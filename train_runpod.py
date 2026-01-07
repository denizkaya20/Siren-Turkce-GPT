import os, sys, json, math, time, shutil
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR    = Path(os.getcwd())
DATA_DIR    = BASE_DIR / "data" / "tr_bin_platinum" 
CHECK_DIR   = BASE_DIR / "checkpoints"
CHECK_DIR.mkdir(exist_ok=True)
RESUME_FROM = "resume_model.pt"
#resume_model.pt, latest.pt dir. latest.pt nin ismini bu sekilde degistirerek modele girdi olarak veriniz

cfg = {
    "n_layer": 24, "n_head": 16, "n_embd": 1024, "block_size": 1024, "vocab_size": 32000,
    "batch_size": 8, "grad_accum": 8, 
    "lr": 2e-5, "min_lr": 1e-6, "warmup_steps": 100, "max_steps": 30000,
    "compile": True, "dtype": "float16", 
    "eval_interval": 1000, "patience": 5, "min_delta": 0.001,
    "dropout": 0.1       
}

device = "cuda"
torch.set_float32_matmul_precision("high")

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(cfg["n_embd"]),
                "attn": nn.MultiheadAttention(cfg["n_embd"], cfg["n_head"], batch_first=True, dropout=cfg["dropout"]),
                "ln2": nn.LayerNorm(cfg["n_embd"]),
                "mlp": nn.Sequential(
                    nn.Linear(cfg["n_embd"], 4*cfg["n_embd"]), nn.GELU(),
                    nn.Linear(4*cfg["n_embd"], cfg["n_embd"]), nn.Dropout(cfg["dropout"])
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
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        for b in self.blocks:
            x_ = b.ln1(x)
            x = x + b.attn(x_, x_, x_, attn_mask=mask, is_causal=True)[0]
            x = x + b.mlp(b.ln2(x))
        x = self.ln_f(x)
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return self.head(x), None

train_data = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap(DATA_DIR / "val.bin",   dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg["block_size"], (cfg["batch_size"],))
    x = torch.stack([torch.from_numpy(data[i:i+cfg["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+cfg["block_size"]].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def get_lr(step):
    if step < cfg["warmup_steps"]: return cfg["lr"] * step / cfg["warmup_steps"]
    if step > cfg["max_steps"]: return cfg["min_lr"]
    decay_ratio = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])

print(f"🚀 Model Hazırlanıyor...")
model = GPT(cfg).to(device)

if os.path.exists(RESUME_FROM):
    print(f"🔄 RESUME: {RESUME_FROM} yükleniyor...")
    print("resume_model.pt yuklenmediyse latest.pt yi bu isimle degistirip klasore kaydediniz")
    ckpt = torch.load(RESUME_FROM, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    new_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    print("✅ Model ağırlıkları yüklendi!")
else:
    print("❌ HATA: Resume dosyası yok!")
    sys.exit(1)

if cfg["compile"]: model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=(cfg["dtype"]=="float16"))

step = 0
t0 = time.time()
best_val_loss = float('inf')
patience = 0

print(f"🔥 Platinum Eğitim Başlıyor! (Hedef: {cfg['max_steps']} adım)")

while step < cfg["max_steps"]:
    model.train()
    opt.zero_grad()
    loss_acc = 0
    for _ in range(cfg["grad_accum"]):
        x, y = get_batch("train")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
            loss /= cfg["grad_accum"]
        scaler.scale(loss).backward()
        loss_acc += loss.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for p in opt.param_groups: p["lr"] = lr
    scaler.step(opt)
    scaler.update()
    step += 1
    
    if step % 10 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"Step {step:5d} | Loss: {loss_acc * cfg['grad_accum']:.4f}")
        sys.stdout.flush()

    if step % cfg["eval_interval"] == 0:
        print(f"\n🔍 Değerlendirme...")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(20):
                x, y = get_batch("val")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"📊 Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")
        
        path = CHECK_DIR / f"platinum_step_{step}.pt"
        torch.save(model.state_dict(), path)
        print(f"💾 Checkpoint kaydedildi: {path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_path = CHECK_DIR / "platinum_BEST.pt"
            torch.save(model.state_dict(), best_path)
            print(f"⭐ En iyi model güncellendi: {best_path}")
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("🛑 Erken Durdurma!")
                break

print("\n🏁 BİTTİ!")
