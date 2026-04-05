import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Ortam Değişkenleri
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

app = FastAPI(title="Kurumsal Bilgi Asistanı", description="Corporate Knowledge Assistant Backend Service")

# CORS ayarları (Next.js frontend'in bu API ile konuşabilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Geliştirme aşamasında her yerden erişime izin veriyoruz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. RAG (Retrieval-Augmented Generation) Hazırlığı
print("Modeller ve Vektör Veritabanı yükleniyor...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Vektör veritabanı root dizininde olduğu için ona göre yol veriyoruz
persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

if not os.path.exists(persist_directory):
    print(f"UYARI: Vektör veritabanı {persist_directory} adresinde bulunamadı!")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# Groq LLM (Llama 3.3)
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )
except Exception as e:
    print(f"HATA: Groq bağlantısı kurulamadı: {e}")
    llm = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt Şablonu
template = """Sen İstanbul Gedik Üniversitesi'nin resmi kurumsal bilgi asistanısın. 
Aşağıdaki bağlamı (context) kullanarak, üniversite yönetmelikleri, sınav kuralları ve idari prosedürler hakkındaki kullanıcı sorusunu yanıtla.

HAREKET TARZI:
1. SADECE sana verilen bağlamdaki bilgilere sadık kal. Kendi bilgilerini veya genel bilgileri katma.
2. Eğer cevap bağlamda yoksa, hayal kurma; "Maalesef bu konuda üniversite yönetmeliklerinde net bir bilgi bulamadım, ilgili birimle görüşmenizi öneririm" de.
3. Cevaplarını net, samimi ama akademik ciddiyete uygun bir dille ver.
4. Bilgi bir "Madde" (örn: Madde 33) içeriyorsa, cevabında bu madde numarasını mutlaka belirt.
5. Eğer kullanıcı selam veriyorsa, teşekkür ediyorsa veya iyi dileklerde bulunuyorsa (merhaba, teşekkürler, iyi günler vb.), bu duruma uygun, nazik ve kurumsal bir dille (örn: "Rica ederim, yardımcı olabildiğime sevindim. Başka bir sorunuz olursa beklerim.") karşılık ver.

BAĞLAM:
{context}

SORU:
{question}

CEVAP:"""

prompt = ChatPromptTemplate.from_template(template)

# LCEL Zinciri
if llm:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = None

# 3. API Uç Noktaları (Endpoints)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Kullanıcıdan gelen soruyu alır ve RAG asistanından gelen cevabı döner.
    """
    if not rag_chain:
        raise HTTPException(status_code=500, detail="LLM servisi aktif değil.")
    
    try:
        response = rag_chain.invoke(request.message)
        return ChatResponse(answer=response)
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate_limit" in error_msg:
            # Kullanıcının özel istediği yoğunluk mesajı
            return ChatResponse(answer="Şu an bir yoğunluk var, birkaç dakika içinde tekrar deneyin.")
        
        print(f"Sohbet hatası: {e}")
        raise HTTPException(status_code=500, detail="Sistem bir hata ile karşılaştı. Lütfen daha sonra tekrar deneyin.")

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "message": "NLP RAG API is running",
        "models": {
            "embedding": "paraphrase-multilingual-MiniLM-L12-v2",
            "llm": "llama-3.3-70b-versatile"
        }
    }

if __name__ == "__main__":
    # Geliştirme sunucusunu başlat
    uvicorn.run(app, host="0.0.0.0", port=8000)
