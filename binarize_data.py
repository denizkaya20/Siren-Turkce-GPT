import os, json, sys
from pathlib import Path
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

BASE_DIR    = Path(os.getcwd())
SRC_DIR     = BASE_DIR / "data" / "tr_src"
OUT_DIR     = BASE_DIR / "data" / "tr_bin_platinum"
TOK_DIR     = BASE_DIR / "tokenizer"
CORPUS_FILE = SRC_DIR / "platinum_training_corpus.txt"
TOK_MODEL   = TOK_DIR / "tr_unigram32k.model"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CORPUS_FILE.exists() or not TOK_MODEL.exists():
    print("❌ HATA: Dosyalar eksik!")
    sys.exit(1)

sp = spm.SentencePieceProcessor()
sp.Load(str(TOK_MODEL))
EOS_ID = sp.eos_id()
VOCAB_SIZE = sp.GetPieceSize()

print(f"🧠 Tokenizer: {VOCAB_SIZE}")
lines = sum(1 for _ in open(CORPUS_FILE, "r", encoding="utf-8"))

train_path = OUT_DIR / "train.bin"
val_path   = OUT_DIR / "val.bin"
dtype = np.uint16

with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=lines)):
            line = line.strip()
            if not line: continue
            ids = sp.Encode(line, out_type=int) + [EOS_ID]
            if i % 20 == 0: f_val.write(np.array(ids, dtype=dtype).tobytes())
            else: f_train.write(np.array(ids, dtype=dtype).tobytes())

with open(OUT_DIR / "meta.json", "w") as f:
    json.dump({"vocab_size": VOCAB_SIZE, "dtype": "uint16"}, f)
print("✅ Binarize Bitti!")
