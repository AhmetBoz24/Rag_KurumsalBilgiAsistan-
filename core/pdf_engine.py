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
    pdf_path = BASE_DIR / "data" / "58336.pdf"
    
    if not pdf_path.exists():
        print(f"Hata: {pdf_path} dosyası bulunamadı!")
        return
    
    print(f"İşleniyor: {pdf_path.name}...")
    text = extract_text_from_pdf(pdf_path)
    
    if text and len(text) > 100:  # En azından anlamlı bir metin olmalı
        output_path = BASE_DIR / "data" / "corpus.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"✓ BAŞARILI: Tam metin '{output_path}' dosyasına kaydedildi.")
        print(f"Toplam karakter sayısı: {len(text)}")
        
        # Kritik veri kontrolü
        if "MADDE 33" in text.upper():
            print("✓ Doğrulama: MADDE 33 metin içerisinde bulundu.")
        else:
            print("⚠ UYARI: MADDE 33 metin içerisinde bulunamadı! Lütfen PDF'i kontrol edin.")
    else:
        print("Hata: PDF'den yeterli metin çıkarılamadı.")

if __name__ == "__main__":
    main()
