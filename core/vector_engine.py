import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core.chunkers.agentic import run as agentic_chunk

def build_vector_store(corpus_path: str, persist_directory: str):
    """
    Corpus dosyasını okur, AGENTIC Chunker ile anlamlı parçalara böler 
    ve ChromaDB'ye kaydedir.
    """
    if not os.path.exists(corpus_path):
        print(f"Hata: {corpus_path} bulunamadı. Lütfen önce pdf_engine.py çalıştırın.")
        return

    print(f"Metin okunuyor: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Gedik Üniversitesi için AGENTIC (Başlık/Madde Duyarlı) Parçalama
    print("Akıllı Hafıza (Agentic Chunking) uygulanıyor...")
    chunks = agentic_chunk(text, target_chars=1000, overlap_sent=1)
    print(f"✓ {len(chunks)} anlamlı parça oluşturuldu.")

    print("Embedding modeli (paraphrase-multilingual-MiniLM-L12-v2) yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Vektör veritabanı güncelleniyor: {persist_directory}")
    # Eğer varsa eski db'yi temizlemek daha garanti olur (üretim temizliği)
    import shutil
    if os.path.exists(persist_directory):
        print("Eski veritabanı temizleniyor...")
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("✓ İşlem başarıyla tamamlandı. Hafıza güncellendi.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(BASE_DIR, "data", "corpus.txt")
    db_path = os.path.join(BASE_DIR, "chroma_db")
    build_vector_store(corpus, db_path)
