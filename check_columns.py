from datasets import load_dataset

print("🔍 COSMOS Veri Seti İnceleniyor...")
try:
    # Streaming olmadan sadece ilk örneği çekelim
    ds = load_dataset("Berkesule/COSMOS-Sentetic-Turkish-Corpus-2GB-Clean", split="train", streaming=True)
    
    print("\n✅ Bağlantı Başarılı! İlk örneğin anahtarları (sütunları):")
    for item in ds:
        print(item.keys())
        print("\nÖrnek İçerik (Kısaltılmış):")
        print(str(item)[:200])
        break
except Exception as e:
    print(f"\n❌ Hata: {e}")
