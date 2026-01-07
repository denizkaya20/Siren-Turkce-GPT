import os
import numpy as np
import sentencepiece as spm # Tiktoken yerine SentencePiece
from datasets import load_dataset
from tqdm import tqdm
import json
import html
import unicodedata

# -----------------------------------------------------------------------------
# AYARLAR
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'platinum_tr')
TOKENIZER_MODEL = 'tokenizer/tr_unigram32k.model' # Senin model dosyan

os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 1. Tokenizer'ı Yükle
if not os.path.exists(TOKENIZER_MODEL):
    raise FileNotFoundError(f"HATA: {TOKENIZER_MODEL} dosyası bulunamadı! Lütfen script ile aynı klasöre koy.")

print(f"📖 Tokenizer yükleniyor: {TOKENIZER_MODEL}")
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
vocab_size = sp.get_piece_size()
print(f"   Vocab Size: {vocab_size}")

# EOT (End of Text) Token belirlemesi
# Genelde SentencePiece'de EOS id kullanılır (genelde 2 veya 1'dir)
EOT_TOKEN = sp.eos_id() 
if EOT_TOKEN == -1: EOT_TOKEN = 2 # Eğer tanımlı değilse manuel atanır

def clean_text(text):
    if not text: return ""
    # HTML karakterlerini düzelt
    text = html.unescape(str(text)).replace("\xa0", " ")
    # Unicode normalizasyonu
    text = unicodedata.normalize("NFKC", text)
    return text.strip()

# Veri Toplayıcı Liste
all_token_ids = []

# --- 1. COSMOS DATASET (HuggingFace) ---
print("📥 COSMOS Dataset İndiriliyor...")
try:
    ds_cosmos = load_dataset("Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB-Clean", split="train", streaming=True)
    for item in tqdm(ds_cosmos, desc="COSMOS Tokenize"):
        text = clean_text(item.get('text', ''))
        if len(text) > 20:
            # SENTENCEPIECE ile tokenize et
            tokens = sp.EncodeAsIds(text)
            all_token_ids.extend(tokens)
            all_token_ids.append(EOT_TOKEN) 
except Exception as e:
    print(f"❌ Hata (COSMOS): {e}")

# --- 2. MY GOLD DATASET (Önceki sentetik verin) ---
gold_path = "my_gold_dataset.jsonl"
if os.path.exists(gold_path):
    print("📥 Gold Dataset İşleniyor...")
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Gold Data"):
            try:
                d = json.loads(line)
                # Soru-Cevap formatını metne döküyoruz
                text = f"Soru: {clean_text(d.get('instruction',''))}\nCevap: {clean_text(d.get('output',''))}"
                tokens = sp.EncodeAsIds(text)
                all_token_ids.extend(tokens)
                all_token_ids.append(EOT_TOKEN)
            except: pass

# --- 3. WIKI QA ---
try:
    print("📥 Wiki QA İşleniyor...")
    ds_wiki = load_dataset("avometre/turkish-wikipedia-qa", split="train", streaming=True)
    for item in tqdm(ds_wiki, desc="Wiki QA"):
        q = clean_text(item.get('question', ''))
        a = clean_text(item.get('answer', ''))
        text = f"Soru: {q}\nCevap: {a}"
        all_token_ids.extend(sp.EncodeAsIds(text))
        all_token_ids.append(EOT_TOKEN)
except: print("❌ Hata (Wiki)")

# --- 4. ALPACA ---
try:
    print("📥 Alpaca İşleniyor...")
    ds_alpaca = load_dataset("TFLai/Turkish-Alpaca", split="train", streaming=True)
    for item in tqdm(ds_alpaca, desc="Alpaca"):
        q = clean_text(item.get('instruction', ''))
        a = clean_text(item.get('output', ''))
        text = f"Soru: {q}\nCevap: {a}"
        all_token_ids.extend(sp.EncodeAsIds(text))
        all_token_ids.append(EOT_TOKEN)
except: print("❌ Hata (Alpaca)")


# --- KAYIT İŞLEMİ (.bin dosyaları) ---
total_tokens = len(all_token_ids)
print(f"📊 Toplam Token Sayısı: {total_tokens:,}")
print("💾 Binary dosyalara yazılıyor...")

# uint16: 0-65535 arası sayı tutar.
all_tokens_np = np.array(all_token_ids, dtype=np.uint16)

# Train / Val Split
# Knowledge injectionda verinin çoğunu train'e ayırırız.
n = len(all_tokens_np)
train_data = all_tokens_np[:int(n*0.95)]
val_data = all_tokens_np[int(n*0.95):]

# Dosyaları yaz
train_data.tofile(os.path.join(DATA_CACHE_DIR, 'train.bin'))
val_data.tofile(os.path.join(DATA_CACHE_DIR, 'val.bin'))

# Meta dosyasını da (vocab size vb.) kaydedelim ki train.py okuyabilsin
meta = {
    'vocab_size': vocab_size,
    'tokenizer': 'sentencepiece',
    'model_file': TOKENIZER_MODEL
}
import pickle
with open(os.path.join(DATA_CACHE_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"✅ Hazır! Dosyalar burada: {DATA_CACHE_DIR}")
print(f"   Train tokens: {len(train_data):,}")
print(f"   Val tokens: {len(val_data):,}")
