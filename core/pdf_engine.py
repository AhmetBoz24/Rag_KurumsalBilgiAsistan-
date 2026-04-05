import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """
    PyMuPDF kullanarak PDF dosyasındaki metni her sayfayı gezerek ayıklar.
    Daha yüksek doğruluk ve hız için fitz tercih edilmiştir.
    """
    all_text = []
    try:
        doc = fitz.open(pdf_path)
        print(f"PDF Açıldı: {pdf_path}")
        print(f"Toplam Sayfa Sayısı: {len(doc)}")
        
        for i, page in enumerate(doc):
            # "text" modu hiyerarşiyi ve madde numaralarını korumak için iyidir
            text = page.get_text("text")
            if text:
                all_text.append(f"--- SAYFA {i+1} ---\n{text}")
            else:
                print(f"Uyarı: Sayfa {i+1} boş görünüyor.")
        
        doc.close()
        return "\n\n".join(all_text)
    except Exception as e:
        print(f"PDF okuma hatası (PyMuPDF): {e}")
        return ""

def main():
    BASE_DIR = Path(__file__).parent.parent
    data_dir = BASE_DIR / "data"
    
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Hata: {data_dir} klasöründe hiç PDF dosyası bulunamadı!")
        return
    
    print(f"Toplam {len(pdf_files)} adet PDF bulundu. İşleniyor...")
    
    all_combined_text = ""
    
    for pdf_path in pdf_files:
        print(f"\n--- İşleniyor: {pdf_path.name} ---")
        text = extract_text_from_pdf(pdf_path)
        
        if text and len(text) > 100:
            all_combined_text += f"\n\n==== DOKÜMAN BAŞLANGICI: {pdf_path.name} ====\n\n"
            all_combined_text += text
        else:
            print(f"⚠ Uyarı: {pdf_path.name} dosyasından yeterli metin çıkarılamadı.")

    if len(all_combined_text) > 100:
        output_path = data_dir / "corpus.txt"
        output_path.write_text(all_combined_text, encoding="utf-8")
        print(f"\n✓ BAŞARILI: Tüm PDF'ler birleştirildi ve '{output_path.name}' dosyasına kaydedildi.")
        print(f"Toplam karakter sayısı: {len(all_combined_text)}")
    else:
        print("Hata: PDF'lerden yeterli metin çıkarılamadı.")

if __name__ == "__main__":
    main()
