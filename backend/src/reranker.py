"""
reranker.py  —  LawAgent Reranker (v2-Production)
--------------------------------------------------
Özellikler:
  - CrossEncoder tabanlı semantic reranking
  - Türkçe hukuki sorgu tipi sınıflandırması (classify_query)
  - Lazy loading: model ilk kullanımda yüklenir
  - Skor normalizasyonu: sigmoid ile [0,1] aralığına çekilir

CrossEncoder seçimi:
  - "cross-encoder/ms-marco-MiniLM-L-6-v2"  → hız odaklı, ~85 MB
  - "cross-encoder/ms-marco-MiniLM-L-12-v2" → kalite odaklı, ~130 MB
  Türkçe corpus için ms-marco modeller çapraz dil transferi ile çalışır.
  Daha iyi sonuç için: "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
"""

import math
import os
from functools import lru_cache

from sentence_transformers import CrossEncoder

# ─── SABITLER ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Türkçe hukuki intent anahtar kelimeleri
_MEVZUAT_KEYS   = ["kaç", "süresi", "madde", "kaç gün", "süre", "tarih", "şart", "koşul"]
_YARGITAY_KEYS  = ["yargıtay", "karar", "emsal", "içtihat", "daire", "bozma"]
_ACIKLAMA_KEYS  = ["nasıl", "hesaplanır", "nedir", "ne demek", "tanımı", "açıkla"]
_KARSILASTIRMA_KEYS = ["fark", "arasındaki", "mi yoksa", "hangisi", "ile arasında"]


# ─── QUERY CLASSIFIER ─────────────────────────────────────────────────────────

def classify_query(sorgu: str) -> str:
    """
    Türkçe hukuki sorguyu 5 tipten birine sınıflandırır.

    Dönüş değerleri:
        "mevzuat"      → Kanun maddesi / süre / şart sorusu
        "yargitay"     → Yargıtay kararı / emsal arama
        "aciklama"     → Kavram açıklaması / tanım
        "karsilastirma"→ İki kavram / durum karşılaştırması
        "genel"        → Diğer
    """
    s = sorgu.lower()

    if any(k in s for k in _KARSILASTIRMA_KEYS):
        return "karsilastirma"
    if any(k in s for k in _YARGITAY_KEYS):
        return "yargitay"
    if any(k in s for k in _MEVZUAT_KEYS):
        return "mevzuat"
    if any(k in s for k in _ACIKLAMA_KEYS):
        return "aciklama"
    return "genel"


# ─── RERANKER ─────────────────────────────────────────────────────────────────

class LegalReranker:
    """
    CrossEncoder tabanlı reranker.

    Kullanım:
        reranker = LegalReranker()
        ranked   = reranker.rerank(query, chunks, top_k=5)

    Her chunk dict'inde "rerank_skor" (float, 0-1) alanı eklenir.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None  # lazy load

    # ------------------------------------------------------------------
    # Lazy loading: modeli ilk rerank çağrısında yükle
    # ------------------------------------------------------------------

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[Reranker] CrossEncoder yükleniyor: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Ham CrossEncoder skorunu [0, 1] aralığına normalize eder."""
        return 1.0 / (1.0 + math.exp(-x))

    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Verilen chunk listesini CrossEncoder ile yeniden sıralar.

        Args:
            query  : Kullanıcı sorgusu
            chunks : retrieve() çıktısı — "text" alanı zorunlu
            top_k  : Döndürülecek maksimum sonuç sayısı

        Returns:
            rerank_skor eklenerek sıralanmış chunk listesi (en fazla top_k eleman)
        """
        if not chunks:
            return []

        pairs  = [(query, c["text"]) for c in chunks]
        raw_scores = self.model.predict(pairs)

        for c, raw in zip(chunks, raw_scores):
            c["rerank_skor"] = round(self._sigmoid(float(raw)), 6)

        chunks.sort(key=lambda x: -x["rerank_skor"])
        return chunks[:top_k]