"""
generator.py — LawAgent AI Backend (v5.0 - Geliştirilmiş Sürüm)
=================================================================
Proje: TÜBİTAK 2209/A

YENİ ÖZELLİKLER (v5.0):
  1. Session-based Conversation Memory (Son 4 mesaj geçmişi)
  2. Hallüsinasyon Kontrolü (Self-Correction & Source Validation)
  3. Dinamik K Retrieval (Intent-based precision adjustment)
  4. Query Intent Router (Bilgi Alma, Karşılaştırma, Taslak vb.)
  5. Gelişmiş hata yönetimi ve logging
"""

import os
import re
import time
import argparse
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq, APIStatusError, APITimeoutError, RateLimitError
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from retriever import LegalRetriever

# ─── LOGGING ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LawAgent.Generator.v5")


# ─── ENV ────────────────────────────────────────────────────────────────────

_ENV_ADAYLARI = [
    Path("/content/drive/MyDrive/lawagent/.env"),
    Path(__file__).resolve().parent.parent.parent / ".env",
    Path(__file__).resolve().parent / ".env",
]
for env_path in _ENV_ADAYLARI:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        log.info(f".env yüklendi: {env_path}")
        break

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY bulunamadı! .env dosyasını kontrol et.")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SESSION MEMORY (Conversation Buffer)
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Session tabanlı sohbet geçmişi yönetimi."""
    
    def __init__(self, max_memory: int = 4):
        """
        Args:
            max_memory: Tutulacak maksimum mesaj çifti sayısı (user + assistant)
        """
        self.memory: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.max_memory = max_memory
    
    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str):
        """Kullanıcı-asistan mesaj çiftini belleğe ekle."""
        if session_id not in self.memory:
            self.memory[session_id] = []
        
        self.memory[session_id].append({
            "role": "user",
            "content": user_msg,
            "timestamp": datetime.now().isoformat()
        })
        self.memory[session_id].append({
            "role": "assistant",
            "content": assistant_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        # Son max_memory*2 mesajı tut (user+assistant çiftleri)
        if len(self.memory[session_id]) > self.max_memory * 2:
            self.memory[session_id] = self.memory[session_id][-(self.max_memory * 2):]
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Seçilen session'ın geçmişini döndür."""
        return self.memory.get(session_id, [])
    
    def get_context_string(self, session_id: str) -> str:
        """Prompt'a eklenebilir context string formatında geçmiş döndür."""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        context_lines = ["--- ÖNCEKI BAĞLAM ---"]
        for msg in history[-4:]:  # Son 2 döngü (4 mesaj)
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            context_lines.append(f"{role}: {msg['content'][:300]}")  # 300 char limit
        
        return "\n".join(context_lines) + "\n\n"


# ─── HUKUKI FİLTRE ──────────────────────────────────────────────────────────

_HUKUK_DISI = {
    "hava", "yemek", "müzik", "film", "spor", "oyun",
    "minecraft", "magazin", "haber", "gündem", "sağlık",
    "doktor", "ilaç", "matematik", "fizik", "kimya",
}

_HUKUKI_SINYALLER = {
    "nedir", "nasıl", "hak", "kanun", "madde", "dava",
    "sözleşme", "tazminat", "kira", "borç", "alacak",
    "fesih", "temerrüt", "cayma", "garanti", "tahliye",
    "tbk", "tkhk", "ttk", "6098", "6502", "6102",
    "mahkeme", "icra", "ipotek", "miras", "velayet",
}


def is_legal_query(sorgu: str) -> bool:
    s = sorgu.lower()
    if any(hd in s.split() for hd in _HUKUK_DISI):
        return False
    return (
        any(sig in s for sig in _HUKUKI_SINYALLER)
        or len(sorgu.split()) >= 3
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUERY INTENT ROUTER (Niyet Analizi)
# ═══════════════════════════════════════════════════════════════════════════════

class QueryIntentRouter:
    """Kullanıcı sorusunun niyet/amacını belirler."""
    
    INTENT_DEFINITIONS = {
        "INFO_RETRIEVAL": {
            "keywords": ["nedir", "ne", "neyin", "nasıl", "hangi", "kaç"],
            "description": "Genel bilgi alma",
            "retrieval_k": 7,
        },
        "COMPARISON": {
            "keywords": ["fark", "arasında", "farklı", "ne kadar", "vs", "karşılaştır"],
            "description": "İki veya daha fazla madde/kavram karşılaştırması",
            "retrieval_k": 10,
        },
        "PROCEDURE": {
            "keywords": ["süre", "yapılır", "adım", "işlem", "başvuru", "başvur"],
            "description": "Yasal bir prosedürün adımları/süresi",
            "retrieval_k": 8,
        },
        "RIGHTS_OBLIGATION": {
            "keywords": ["hak", "sorumluluk", "yükümlülük", "ödeme", "iade"],
            "description": "Haklar ve yükümlülükler",
            "retrieval_k": 7,
        },
        "CONSEQUENCE": {
            "keywords": ["sonuç", "ceza", "para", "tazminat", "zarar", "risiko"],
            "description": "Hukuki bir eyleminin sonuçları",
            "retrieval_k": 6,
        },
    }
    
    def __init__(self, client: Groq):
        self.client = client
    
    def detect_intent(self, sorgu: str) -> Tuple[str, int]:
        """
        Sorunun niyet ve uygun K değerini döndür.
        
        Returns:
            (intent_type, recommended_k)
        """
        sorgu_lower = sorgu.lower()
        
        # Anahtar kelime eşleştirmesi
        best_intent = "INFO_RETRIEVAL"
        best_score = 0
        
        for intent, config in self.INTENT_DEFINITIONS.items():
            score = sum(1 for kw in config["keywords"] if kw in sorgu_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        recommended_k = self.INTENT_DEFINITIONS[best_intent]["retrieval_k"]
        
        log.info(f"Intent Detection: {best_intent} (k={recommended_k}), sorgu: '{sorgu[:50]}'")
        
        return best_intent, recommended_k


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HALLÜSINASYON KONTROLÜ (Self-Correction)
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationValidator:
    """Üretilen yanıtın kaynakla tutarlılığını kontrol eder."""
    
    # Madde referans regex
    _MADDE_REF_PATTERN = re.compile(
        r"(\w+)?\s*m(?:adde)?\.?\s*(\d+(?:\s*(?:/[a-z])?)?)\b",
        re.IGNORECASE
    )
    
    def __init__(self, client: Groq):
        self.client = client
    
    @staticmethod
    def extract_article_refs(text: str) -> List[str]:
        """Metinden madde referanslarını çıkart."""
        refs = []
        for match in HallucinationValidator._MADDE_REF_PATTERN.finditer(text):
            refs.append(match.group(0).strip())
        return refs
    
    @staticmethod
    def extract_source_articles(chunks: List[Dict]) -> List[str]:
        """Kaynaklar listesinden madde numaralarını çıkart."""
        articles = []
        for chunk in chunks:
            article_no = chunk.get("article_no", "")
            if article_no:
                articles.append(str(article_no).strip())
        return articles
    
    def validate_faithfulness(
        self, 
        answer: str, 
        chunks: List[Dict], 
        threshold: float = 0.7
    ) -> Tuple[bool, str, List[str]]:
        """
        Yanıtın kaynakla tutarlı olup olmadığını kontrol et.
        
        Returns:
            (is_faithful, warning_message, mentioned_refs)
        """
        mentioned_refs = self.extract_article_refs(answer)
        source_articles = self.extract_source_articles(chunks)
        
        if not mentioned_refs:
            # Madde referansı yok ama kaynaklar varsa uyarı
            if chunks:
                warning = (
                    "⚠️ Not: Yanıtta madde numaraları belirtilmemiş. "
                    "Lütfen doğruluk için kaynak maddeleri kontrol edin."
                )
                return True, warning, []
            return True, "", []
        
        # Belirtilen maddelerin kaynakta olup olmadığını kontrol et
        valid_refs = [ref for ref in mentioned_refs if any(src in ref for src in source_articles)]
        
        if not valid_refs and mentioned_refs:
            warning = (
                f"⚠️ HALLÜSİNASYON RİSKİ UYARISI: "
                f"Yanıtta belirtilen madde(ler) kaynaklarda doğrudan yer almayabilir. "
                f"Bu bilgileri doğrulamak için lütfen resmi kaynaklara başvurun."
            )
            return False, warning, mentioned_refs
        
        return True, "", mentioned_refs


# ─── QUERY REWRITE ──────────────────────────────────────────────────────────

_MADDE_REF_RE = re.compile(
    r"\b(tbk|tkhk|ttk)\s*(?:m\.|madde)?\s*\d+\b"
    r"|\b(6098|6502|6102)\b"
    r"|\b(?:madde|m\.)\s*\d+\b",
    re.IGNORECASE,
)

_REWRITE_SYSTEM = (
    "Sen Türk hukuku uzmanısın. Kullanıcının sorusunu, "
    "anlamını bozmadan akademik hukuk terimleriyle yeniden yaz. "
    "Kanun kısaltmalarını (TBK, TKHK, TTK) koru. "
    "Sadece yeniden yazılmış soruyu döndür, açıklama ekleme."
)


def has_madde_ref(sorgu: str) -> bool:
    return bool(_MADDE_REF_RE.search(sorgu))


def rewrite_query(client: Groq, sorgu: str) -> str:
    """Soruyu akademik hukuk terimleriyle yeniden yaz."""
    if has_madde_ref(sorgu):
        return sorgu

    if len(sorgu.split()) < 4 or len(sorgu.split()) > 30:
        return sorgu

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user",   "content": f"Soru: {sorgu}\n\nYeniden yazılmış hali:"},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        yeni = resp.choices[0].message.content.strip()
        if not yeni or len(yeni) > 300:
            return sorgu
        return yeni
    except Exception as e:
        log.warning(f"Query rewrite hatası: {e}")
        return sorgu


# ─── GELIŞMIŞ SISTEM PROMPT ─────────────────────────────────────────────────

_SISTEM_PROMPT_TEMPLATE = """Sen yapay zeka tabanlı profesyonel bir Türk Hukuk Asistanısın.

BAĞLAM: {context}

GÖREVLERİN:
1. Kullanıcının sorusu netse, SADECE verilen kaynaklara dayanarak kesin ve kaynak göstererek cevap ver.
2. Eğer soru çok kısa veya belirsiz ise, nazikçe "Tam olarak neyi öğrenmek istediğinizi anlayamadım" diyerek 3 örnek soru öner.
3. Hukuk dışı konularda cevap verme, kullanıcıyı hukuki konulara yönlendir.
4. HER ZAMAN: Konuya özel takip soruları öner.

YANIT FORMATI:
**Hukuki Değerlendirme**
[Cevap metni]

**Dayanak Maddeler**
- [Kanun] m.[No]: [Madde içeriği]

**Sizin için önerilerim:**
- [Takip Sorusu 1]
- [Takip Sorusu 2]

GÖZLEMLER:
- Yanıtta belirttiğin madde numaraları KESINLIKLE kaynaklarda bulunmalıdır.
- Emin olmadığın şeyi söyleme; bunun yerine kullanıcıyı resmi kaynaklara yönlendir.
"""


def build_context(chunks: list) -> str:
    """Kaynakları yapılandırılmış format içine koy."""
    satirlar = []
    for i, c in enumerate(chunks, 1):
        satirlar.append(
            f"--- KAYNAK {i} ---\n"
            f"KANUN: {c.get('law', '?')}\n"
            f"MADDE: {c.get('article_no', '?')}\n"
            f"METİN: {c.get('text', '')}"
        )
    return "\n\n".join(satirlar)


# ─── SINGLETON RETRIEVER ────────────────────────────────────────────────────

_retriever_instance: Optional[LegalRetriever] = None


def get_retriever() -> LegalRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        log.info("[Startup] Retriever yükleniyor...")
        _retriever_instance = LegalRetriever()
        log.info("[Startup] Retriever hazır.")
    return _retriever_instance


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEGAL GENERATOR (Ana Sınıf)
# ═══════════════════════════════════════════════════════════════════════════════

class LegalGenerator:
    """Hukuki sorulara cevap veren AI asistanı."""

    def __init__(self, k: int = 7):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY bulunamadı.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.retriever = get_retriever()
        self.default_k = k
        
        # v5.0 Yeni bileşenler
        self.memory = ConversationMemory(max_memory=4)
        self.intent_router = QueryIntentRouter(self.client)
        self.hallucination_validator = HallucinationValidator(self.client)

    def generate(
        self, 
        sorgu: str, 
        session_id: str = "default",
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Hukuki sorguya yanıt üret.
        
        Args:
            sorgu: Kullanıcının sorusu
            session_id: Sohbet oturumunun ID'si (bellek için)
            k: Retrieval sonuç sayısı (opsiyonel)
        
        Returns:
            Yanıt, kaynaklar ve metadata
        """
        t0 = time.time()
        sorgu_temiz = sorgu.lower().strip()

        # ──── 0. SELAMLAŞMAişaret KONTROLÜ ────
        if sorgu_temiz in ["selam", "merhaba", "sa", "as", "günaydın", "iyi günler"]:
            greeting_answer = (
                "Merhaba! Ben LawAgent AI. Türk Borçlar, Ticaret ve Tüketici Hukuku alanlarında size yardımcı olabilirim.\n\n"
                "**Size nasıl yardımcı olabilirim? Örneğin şunları sorabilirsiniz:**\n"
                "- 'Kira sözleşmemi nasıl feshedebilirim?'\n"
                "- 'İnternetten aldığım ürünü iade edebilir miyim?'\n"
                "- 'Borçlu temerrüdü nedir?'"
            )
            self.memory.add_exchange(session_id, sorgu, greeting_answer)
            return {
                "answer": greeting_answer,
                "sources": [],
                "filtered": False,
                "intent": "GREETING",
                "sure_ms": int((time.time() - t0) * 1000),
            }

        try:
            # ──── 1. HUKUKI FİLTRE ────
            if not is_legal_query(sorgu):
                filtered_answer = (
                    "Ben bir Türk hukuk asistanıyım. "
                    "Lütfen Türk Borçlar, Ticaret veya Tüketici hukukuyla "
                    "ilgili bir soru sorunuz."
                )
                self.memory.add_exchange(session_id, sorgu, filtered_answer)
                return {
                    "answer": filtered_answer,
                    "sources": [],
                    "filtered": True,
                    "sure_ms": int((time.time() - t0) * 1000),
                }

            # ──── 2. INTENT DETECTION (YAKALAMALAR BAŞLA) ────
            intent, recommended_k = self.intent_router.detect_intent(sorgu)
            k = k or recommended_k or self.default_k

            # ──── 3. QUERY REWRITE ────
            yeni_sorgu = rewrite_query(self.client, sorgu)

            # ──── 4. RETRIEVAL (BELLEK BİLGİSİ İLE) ────
            # Önceki bağlamı ekle
            history_context = self.memory.get_context_string(session_id)
            retrieval_sorgu = f"{history_context}{sorgu}".strip() if history_context else sorgu
            
            chunks = self.retriever.retrieve(retrieval_sorgu, k=k)

            # Fallback: Eğer çok az sonuç varsa, orijinal sorguyla da ara
            if len(chunks) < 3 and yeni_sorgu != sorgu:
                ek = self.retriever.retrieve(sorgu, k=k)
                mevcut = {c["chunk_id"] for c in chunks}
                for c in ek:
                    if c["chunk_id"] not in mevcut:
                        chunks.append(c)
                chunks = chunks[:k]

            # ──── 5. BOŞSONUÇ KONTROLÜ ────
            if not chunks:
                no_result_answer = (
                    "Üzgünüm, bu konuyu veri tabanımda tam olarak eşleştiremedim. "
                    "Size daha iyi yardımcı olabilmem için sorunuzu biraz daha detaylandırabilir misiniz?\n\n"
                    "**Şunlardan birini mi öğrenmek istemiştiniz?**\n"
                    "- Kira bedelinin geç ödenmesinin (temerrüt) sonuçları nelerdir?\n"
                    "- Tüketici olarak ayıplı malda seçimlik haklarım nelerdir?\n"
                    "- Bir sözleşmenin feshi için gerekli şartlar nelerdir?"
                )
                self.memory.add_exchange(session_id, sorgu, no_result_answer)
                return {
                    "answer": no_result_answer,
                    "sources": [],
                    "filtered": False,
                    "intent": intent,
                    "query_rewritten": yeni_sorgu,
                    "sure_ms": int((time.time() - t0) * 1000),
                }

            # ──── 6. LLM İLE YANIT ÜRET ────
            context = build_context(chunks)
            sistem_prompt = _SISTEM_PROMPT_TEMPLATE.format(context=context)
            
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sistem_prompt},
                    {"role": "user",   "content": f"SORU: {sorgu}"},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            yanit = resp.choices[0].message.content.strip()

            # ──── 7. HALLÜSİNASYON KONTROLÜ ────
            is_faithful, validation_warning, mentioned_refs = (
                self.hallucination_validator.validate_faithfulness(yanit, chunks)
            )
            
            if not is_faithful:
                yanit = yanit + f"\n\n{validation_warning}"
            
            log.info(
                f"Generate başarılı: "
                f"intent={intent}, k={k}, faithful={is_faithful}, "
                f"sources={len(chunks)}, session={session_id}"
            )

            # ──── 8. BELLEĞE EKLE ────
            self.memory.add_exchange(session_id, sorgu, yanit)

            return {
                "answer": yanit,
                "sources": [
                    {
                        "kanun": c.get("law", ""),
                        "madde": c.get("article_no", ""),
                        "ozet": c.get("text", "")[:200] + "...",
                    }
                    for c in chunks
                ],
                "intent": intent,
                "query_rewritten": yeni_sorgu if yeni_sorgu != sorgu else None,
                "hallucination_check": {
                    "is_faithful": is_faithful,
                    "warning": validation_warning,
                    "mentioned_articles": mentioned_refs,
                },
                "sure_ms": int((time.time() - t0) * 1000),
                "filtered": False,
            }

        except RateLimitError:
            log.error("API Rate Limit aşıldı")
            error_answer = "Şu an çok fazla istek alıyorum, lütfen birkaç saniye sonra tekrar deneyin."
            return {"answer": error_answer, "sources": [], "error": "rate_limit"}
        
        except APITimeoutError:
            log.error("API Timeout")
            error_answer = "Sunucu yanıt vermedi, lütfen tekrar deneyin."
            return {"answer": error_answer, "sources": [], "error": "timeout"}
        
        except Exception as e:
            log.exception(f"Kritik Hata: {e}")
            error_answer = "Teknik bir aksaklık oluştu. Lütfen tekrar deneyin."
            return {"answer": error_answer, "sources": [], "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FASTAPI ENTEGRASYON
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()
    log.info("[Startup] Uygulama başladı")
    yield
    global _retriever_instance
    if _retriever_instance and hasattr(_retriever_instance, "qdrant"):
        _retriever_instance.qdrant.close()
    log.info("[Shutdown] Uygulama kapatıldı")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LawAgent AI API",
        version="5.0",
        description="Türk Hukuku Asistanı (Memory, Hallucination Control, Intent Routing)",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def handle_options(request: Request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        return await call_next(request)

    class AskRequest(BaseModel):
        query: str
        k: int = 7
        session_id: str = "default"

    class AskResponse(BaseModel):
        answer: str
        sources: List[Dict[str, str]]
        intent: Optional[str] = None
        query_rewritten: Optional[str] = None
        hallucination_check: Optional[Dict] = None
        sure_ms: int = 0
        filtered: bool = False

    gen = LegalGenerator()

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        """Hukuki sorguya yanıt ver (Memory + Hallucination Check destekli)."""
        if not req.query.strip():
            return JSONResponse(status_code=400, content={"detail": "Sorgu boş."})
        
        result = gen.generate(req.query, session_id=req.session_id, k=req.k)
        return result

    @app.post("/ask-stream")
    async def ask_stream(req: AskRequest):
        """
        Streaming yanıt endpoint (Opsiyonel).
        Frontend'de WebSocket kullanabilirsin.
        """
        if not req.query.strip():
            return JSONResponse(status_code=400, content={"detail": "Sorgu boş."})
        
        result = gen.generate(req.query, session_id=req.session_id, k=req.k)
        return result

    @app.get("/health")
    async def health():
        """Sağlık kontrolü."""
        return {
            "status": "ok",
            "version": "5.0",
            "features": [
                "Conversation Memory",
                "Hallucination Control",
                "Intent Router",
                "Dynamic K Retrieval"
            ]
        }

    @app.get("/memory/{session_id}")
    async def get_memory(session_id: str):
        """Belirli session'ın sohbet geçmişini döndür."""
        history = gen.memory.get_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "history": history
        }

    return app


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="FastAPI sunucusu başlat")
    parser.add_argument("--interactive", action="store_true", help="İnteraktif CLI mod")
    args = parser.parse_args()

    if args.api:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    elif args.interactive:
        # CLI Test Modu
        gen = LegalGenerator()
        session = "cli_session"
        
        print("\n" + "="*70)
        print("LawAgent AI v5.0 - İnteraktif Test Modu")
        print("="*70)
        print("'quit' yazarak çıkabilirsiniz.\n")
        
        while True:
            sorgu = input("Soru: ").strip()
            if sorgu.lower() in ["quit", "q", "çık"]:
                break
            
            if not sorgu:
                continue
            
            result = gen.generate(sorgu, session_id=session)
            
            print("\n" + "-"*70)
            print(f"[{result.get('intent', 'UNKNOWN')}] Yanıt:\n")
            print(result["answer"])
            
            if result.get("hallucination_check", {}).get("warning"):
                print(f"\n⚠️ {result['hallucination_check']['warning']}")
            
            if result.get("sources"):
                print(f"\n📚 Kaynaklar ({len(result['sources'])} adet):")
                for i, src in enumerate(result["sources"], 1):
                    print(f"  {i}. {src['kanun']} m.{src['madde']}")
            
            print(f"\n⏱️ İşlem Süresi: {result.get('sure_ms', 0)}ms\n")
    
    else:
        parser.print_help()