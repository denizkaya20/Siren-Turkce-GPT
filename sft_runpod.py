import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

# ================= AYARLAR =================
BASE_DIR = Path(os.getcwd())
CHECK_DIR = BASE_DIR / "checkpoints"
TOK_DIR = BASE_DIR / "tokenizer"
OUT_DIR = BASE_DIR / "sft_checkpoints"

# En iyi modeli baz alıyoruz (Pretraining'den gelen)
BASE_MODEL_PATH = CHECK_DIR / "platinum_BEST.pt"
TOKENIZER_PATH = TOK_DIR / "tr_unigram32k.model"

BATCH_SIZE = 4
GRAD_ACCUM = 16   # 4 * 16 = 64 efektif batch size
LR = 1e-5         # SFT için düşük öğrenme oranı
EPOCHS = 1        # Tek epoch genelde yeterlidir
MAX_LEN = 512     # Maksimum sekans uzunluğu

device = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR.mkdir(exist_ok=True)

# ================= 1. TOKENIZER =================
if not TOKENIZER_PATH.exists():
    print(f"❌ HATA: Tokenizer bulunamadı: {TOKENIZER_PATH}")
    sys.exit(1)

sp = spm.SentencePieceProcessor()
sp.Load(str(TOKENIZER_PATH))

EOS_ID = sp.eos_id()
# SentencePiece modelinde pad_id yoksa eos_id kullan
PAD_ID = sp.pad_id() if sp.pad_id() >= 0 else EOS_ID 
VOCAB_SIZE = sp.GetPieceSize()

print(f"🧠 Tokenizer: {VOCAB_SIZE} | PAD: {PAD_ID} | EOS: {EOS_ID}")

# ================= 2. VERİ SETİ: Turkish-Alpaca =================
print("📥 Turkish-Alpaca veri seti yükleniyor...")
try:
    ds_alpaca = load_dataset("TFLai/Turkish-Alpaca", split="train")
    print(f"✅ Turkish-Alpaca örnek sayısı: {len(ds_alpaca)}")
except Exception as e:
    print(f"❌ Veri seti yükleme hatası: {e}")
    sys.exit(1)

class AlpacaSFTDataset(Dataset):
    def __init__(self, alpaca_data, tokenizer, max_len):
        self.alpaca_data = alpaca_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.alpaca_data)

    def __getitem__(self, idx):
        item = self.alpaca_data[idx]
        instr = item["instruction"]
        inp = item.get("input", "")
        out = item["output"]

        # Prompt Formatı: (Instruction + Input -> Output)
        if inp:
            prompt = f"Soru: {instr}\nGirdi: {inp}\nCevap: {out}"
        else:
            prompt = f"Soru: {instr}\nCevap: {out}"

        # Tokenize et ve EOS ekle
        ids = self.tokenizer.Encode(prompt, out_type=int) + [EOS_ID]

        # Kırpma (Truncation)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        
        # Dolgu (Padding)
        padding_len = self.max_len - len(ids)
        if padding_len > 0:
            ids = ids + [PAD_ID] * padding_len

        # Güvenlik kontrolü: vocab sınırları
        ids = [max(0, min(t, VOCAB_SIZE - 1)) for t in ids]
        
        return torch.tensor(ids, dtype=torch.long)

train_dataset = AlpacaSFTDataset(ds_alpaca, sp, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"🔥 SFT Eğitimi İçin Hazır Veri: {len(train_dataset)} satır")

# ================= 3. MODEL MİMARİSİ =================
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(cfg["n_embd"]),
                "attn": nn.MultiheadAttention(cfg["n_embd"], cfg["n_head"], batch_first=True, dropout=cfg.get("dropout", 0.0)),
                "ln2": nn.LayerNorm(cfg["n_embd"]),
                "mlp": nn.Sequential(
                    nn.Linear(cfg["n_embd"], 4*cfg["n_embd"]), nn.GELU(),
                    nn.Linear(4*cfg["n_embd"], cfg["n_embd"]), nn.Dropout(cfg.get("dropout", 0.0))
                )
            }) for _ in range(cfg["n_layer"])
        ])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.wte.weight = self.head.weight 

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        
        # Causal Mask
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        for b in self.blocks:
            x_ = b["ln1"](x)
            attn_out, _ = b["attn"](x_, x_, x_, attn_mask=mask, need_weights=False)
            x = x + attn_out
            x = x + b["mlp"](b["ln2"](x))
            
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.head(x)
            
            # --- DÜZELTME BURADA ---
            # Slicing işlemi (x ve y ayrımı) tensor'ü non-contiguous yaptığı için
            # torch.compile ile view hatası veriyordu. .contiguous() ile çözüyoruz.
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)), 
                targets.contiguous().view(-1), 
                ignore_index=PAD_ID
            )
            return logits, loss
            
        return self.head(x), None

# ================= 4. BASE MODELİ YÜKLE =================
if not BASE_MODEL_PATH.exists():
    print(f"❌ HATA: Base model bulunamadı: {BASE_MODEL_PATH}")
    sys.exit(1)

print(f"🚀 Base Model Yükleniyor: {BASE_MODEL_PATH}")
ckpt = torch.load(BASE_MODEL_PATH, map_location=device)

# Config (Eksikse varsayılanları kullan)
cfg = {
    "n_layer": 24, "n_head": 16, "n_embd": 1024, "block_size": 1024, 
    "vocab_size": VOCAB_SIZE, "dropout": 0.0
}

# Checkpoint içinden state_dict al
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
# _orig_mod prefix temizliği
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model = GPT(cfg).to(device)
model.load_state_dict(new_state_dict, strict=False)
print("✅ Base model ağırlıkları başarıyla yüklendi!")

# ================= 5. EĞİTİM (SFT LOOP) =================
# Hız için compile açık
model = torch.compile(model)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

print(f"🔥 SFT Başlıyor! (Epoch: {EPOCHS}, Batch: {BATCH_SIZE}, Accum: {GRAD_ACCUM})")

model.train()
global_step = 0

for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    optimizer.zero_grad()
    
    for batch_idx, inputs in enumerate(pbar):
        inputs = inputs.to(device) # (B, T)
        
        # Input (x) ve Target (y) ayrımı (Shifted)
        # x: 0...T-1 (Girdi)
        # y: 1...T   (Tahmin edilecek bir sonraki token)
        x = inputs[:, :-1]
        y = inputs[:, 1:]
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            
            # Loss değerini eski haline getirip göster (x GRAD_ACCUM)
            pbar.set_description(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item() * GRAD_ACCUM:.4f}")

# ================= 6. KAYDET =================
final_path = OUT_DIR / "SIREN_Platinum_SFT_Alpaca.pt"

save_obj = {
    "model": model.state_dict(),
    "config": cfg,
    "step": global_step,
    "sft_source": "TFLai/Turkish-Alpaca",
    "base_model": str(BASE_MODEL_PATH)
}

torch.save(save_obj, final_path)
print(f"\n🏁 SFT EĞİTİMİ BİTTİ! Model kaydedildi: {final_path}")
print("🎉 Artık modelin talimatlara uyma (Instruction Following) yeteneği kazandı!")
