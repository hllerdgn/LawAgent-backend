"""
retriever.py — LawAgent Hibrit Retriever
=========================================
Proje: TÜBİTAK 2209/A — Borçlar Hukuku ve Ticari Sözleşmelerin
       Otomatik Yorumlanması İçin RAG Tabanlı Akıllı Hukuk Asistanı

Embedder v10 ile tam uyum:
  - encode_single()  → "query: " prefix'i EMBEDDER içinde ekleniyor
  - embed_corpus()   → "LAW Madde NO: text" formatında embed edildi
  Bu asimetri Mursit bi-encoder yapısını doğru kullanır.

Proje formu Hedef 1 metrikleri:
  Recall@10  ≥ 0.75
  MRR@10     ≥ 0.60
  nDCG@10    ≥ 0.70

Özellikler:
  1. Dense Search       → Qdrant cosine (query: prefix embedder içinde)
  2. BM25Plus           → Bellekte Türkçe sparse search + sorgu genişletme
  3. Hybrid Fusion      → Min-Max normalize + alpha ağırlık
  4. Entity Extraction  → detect_kanun() + extract_madde() (regex/keyword)
  5. Shortcut Injection → Madde tespit edildi ama top-100'de yok → enjekte et
  6. Pickle Cache       → BM25 corpus bellekte, Qdrant scroll bir kez

Kullanım:
    from retriever import LegalRetriever
    r = LegalRetriever()
    sonuclar = r.retrieve("Kiracının temerrüdü TBK m.315")
    # [{"law":"TBK","article_no":"315","text":"...","skor":0.92,...}, ...]

Terminal:
    python retriever.py --sorgu "TBK m.315 nedir?"
    python retriever.py --sorgu "Cayma hakkı süresi" --k 10 --reindex
    python retriever.py --sorgu "Anonim şirket" --debug
"""

import math
import os
import pickle
import re
import logging
import atexit
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from embedder import MursitEmbedder, COLLECTION_NAME, QDRANT_STORAGE

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LawAgent.Retriever")


# ═══════════════════════════════════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    """Tüm hiperparametreler tek yerde — kolayca ablasyon yapılabilir."""

    # Aday havuzu genişliği
    TOP_K_DENSE      : int   = 200   # Qdrant'tan kaç aday
    TOP_K_BM25       : int   = 200   # BM25'ten kaç aday

    # Kullanıcıya dönecek final sonuç sayısı
    FINAL_K          : int   = 10

    # Aynı kanun+madde kombinasyonundan max chunk
    MAX_SAME_ARTICLE : int   = 2

    # Hybrid fusion alpha değerleri (dense ağırlığı)
   
    ALPHA_DEFAULT    : float = 0.60  # genel sorgu
    ALPHA_EXACT      : float = 0.30   # madde no tespit edildi → BM25 ön plana
    ALPHA_SEMANTIC   : float = 0.70   # saf semantik sorgu → dense ön plana

    # Boost — fusion sonrası sıralamayı etkiler
    BOOST_MADDE      : float = 10.0    # madde no tam eşleşme
    BOOST_KANUN      : float = 1.2    # kanun eşleşmesi (hafif)

    # Shortcut: madde tespit edildi ama top-100'de yok → corpus'tan enjekte et
    SHORTCUT_ENABLED : bool  = True
    SHORTCUT_MAX     : int   = 3      # kaç chunk enjekte edilsin

    # Minimum skor eşiği (fusion sonrası)
    MIN_SCORE        : float = 0.001

    # Cache dosya yolu
    CACHE_PATH       : str   = "data/retriever_cache.pkl"


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION — detect_kanun() + extract_madde()
# ═══════════════════════════════════════════════════════════════════════════════

# ── Direkt eşleşme (en yüksek öncelik) ────────────────────────────────────────
_DIREKT: Dict[str, str] = {
    # TBK
    "tbk": "TBK", "türk borçlar kanunu": "TBK",
    "borçlar kanunu": "TBK", "6098": "TBK",
    # TKHK
    "tkhk": "TKHK", "tüketici kanunu": "TKHK",
    "tüketicinin korunması": "TKHK", "6502": "TKHK",
    # TTK
    "ttk": "TTK", "türk ticaret kanunu": "TTK",
    "ticaret kanunu": "TTK", "6102": "TTK",
}

# ── Ağırlıklı anahtar kelimeler ────────────────────────────────────────────────
# (kelime, ağırlık) — çok kelimeli ifadeler daha yüksek ağırlık taşır
_KANUN_KW: Dict[str, List[Tuple[str, int]]] = {
    "TKHK": [
        # Proje formu kapsam: ayıplı mal, cayma hakkı, garanti, mesafeli sözleşmeler, tüketici
        ("tüketici hakem heyeti", 5), ("mesafeli sözleşme", 4), ("ayıplı mal", 4),
        ("garanti belgesi", 4), ("tüketici kredisi", 4), ("konut finansmanı", 5),
        ("konut kredisi", 5), ("devre tatil", 4), ("paket tur", 4), ("kapıdan satış", 4),
        ("iş yeri dışında", 4), ("cayma bildirimi", 4), ("süreli yayın", 4),
        ("promosyon", 3), ("kayıp bedel", 3), ("kaçak bedel", 3),
        ("erken ödeme indirimi", 4), ("abonelik sözleşmesi", 4),
        ("cayma hakkı", 3), ("tüketici uyuşmazlığı", 4),
        ("tüketici", 2), ("6502", 5), ("ayıplı", 2), ("cayma", 2),
        ("hakem heyeti", 3), ("garanti", 1), ("iade", 1),
    ],
    "TTK": [
        # Proje formu kapsam: ticari sözleşmeler
        ("ticari defter", 4), ("rekabet yasağı", 4), ("haksız rekabet", 4),
        ("yönetim kurulu", 4), ("genel kurul", 4), ("anonim şirket", 4),
        ("limited şirket", 4), ("kambiyo senedi", 4), ("ticaret sicili", 4),
        ("özen borcu", 4), ("pay sahipliği", 4), ("tasfiye", 3),
        ("poliçe", 3), ("protesto", 3), ("bono", 3), ("konkordato", 3),
        ("ticaret", 2), ("6102", 5), ("şirket", 1), ("tacir", 2),
        ("iflas", 3), ("sigorta", 2), ("acente", 3),
    ],
    "TBK": [
        # Proje formu kapsam: satım, hizmet, eser, kira, temerrüt, ayıplı ifa
        ("kira sözleşmesi", 4), ("kiraya veren", 4), ("kira bedeli", 4),
        ("temerrüt faizi", 4), ("haksız fiil", 4), ("sebepsiz zenginleşme", 4),
        ("kefalet sözleşmesi", 4), ("vekalet sözleşmesi", 4), ("alt kira", 4),
        ("aşırı yararlanma", 4), ("cezai şart", 4), ("ayıplı ifa", 4),
        ("ifa imkânsızlığı", 4), ("sözleşmenin geçersizliği", 4),
        ("eser sözleşmesi", 4), ("hizmet sözleşmesi", 4), ("satım sözleşmesi", 4),
        ("borçlar", 2), ("6098", 5), ("kira", 2), ("kiracı", 3),
        ("temerrüt", 3), ("kefalet", 3), ("vekalet", 3),
        ("ihtar", 2), ("gabin", 4), ("yanılma", 3), ("hata", 1),
        ("tahliye", 3), ("fesih", 2), ("tazminat", 2), ("borçlu", 1),
        ("zamanaşımı", 2), ("faiz", 2), ("sözleşme", 1),
    ],
}

# ── Kanun numarası eşleştirme ─────────────────────────────────────────────────
_KANUN_NO: Dict[str, str] = {"6098": "TBK", "6502": "TKHK", "6102": "TTK"}

# ── Madde referansı regex — geniş format desteği ──────────────────────────────
# Desteklenen formatlar:
#   "TBK m.315"  / "TBK madde 315" / "TBK 315"
#   "madde 315"  / "m.315"
#   "6098 sayılı Kanun'un 315. maddesi"
#   "Borçlar Kanunu 315. maddesi"
_MADDE_RE = re.compile(
    r"""
    (?:
        # ─────────────────────────────────────────────
        # 1. TBK / TTK / TKHK + madde (çok esnek)
        # Örn: TBK m.315 | TBK 315 | TBK'nın 315. maddesi
        # ─────────────────────────────────────────────
        \b(tbk|ttk|tkhk)
        (?:['’]?[a-zçğıöşü]+)?        # ekler (TBK'nın)
        \s*
        (?:m\.?|madde|md\.?)?        # m., madde, md
        \s*
        (\d{1,4})

    |

        # ─────────────────────────────────────────────
        # 2. Sadece madde (suffix destekli)
        # Örn: 315. madde | 315'inci madde | 315 md
        # ─────────────────────────────────────────────
        \b
        (\d{1,4})
        (?:['’]?(?:inci|ıncı|üncü|uncu))?   # 315'inci
        \.?
        \s*
        (?:madde|m\.?|md\.?)
    
    |

        # ─────────────────────────────────────────────
        # 3. 6098 sayılı kanun + madde
        # ─────────────────────────────────────────────
        \b(6098|6502|6102)
        \s+say[ıi]l[ıi]
        [^.\n]{0,60}?
        (\d{1,4})
        (?:\.?\s*(?:madde|m\.?|md\.?))?

    |

        # ─────────────────────────────────────────────
        # 4. Kanun adı + madde
        # Örn: Borçlar Kanunu 315. madde
        # ─────────────────────────────────────────────
        \b(borçlar|tüketici|ticaret)
        \s+kanunu?
        [^.\n]{0,40}?
        (\d{1,4})
        (?:\.?\s*(?:madde|m\.?|md\.?))?

    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_kanun(query: str) -> Optional[str]:
    """
    Sorgudan kanun adını tespit eder.

    Öncelik sırası:
    1. Direkt eşleşme (kanun adı/numarası) → kesin
    2. Regex madde referansı (TBK m.315)   → çok güvenilir
    3. Ağırlıklı anahtar kelime puanı      → istatistiksel

    Returns:
        "TBK" | "TKHK" | "TTK" | None
    """
    q = query.lower().strip()

    # 1. Direkt eşleşme
    for ifade, kanun in _DIREKT.items():
        if ifade in q:
            log.debug(f"Kanun tespiti (direkt): {kanun}")
            return kanun

def detect_kanun(query: str) -> Optional[str]:
    """
    Sorgudan kanun adını tespit eder. 
    İyileştirme: Belirsizlik durumunda None dönerek genel arama (fallback) sağlar.
    """
    q = query.lower().strip()

    # 1. Direkt eşleşme 
    for ifade, kanun in _DIREKT.items():
        if ifade in q:
            return kanun

    # 2. Regex madde referansı 
    for m in _MADDE_RE.finditer(q):
        g = m.groups()
        if g[0]: return g[0].upper()
        if g[3]:
            kanun = _KANUN_NO.get(g[3])
            if kanun: return kanun

    # 3. Ağırlıklı kelime puanı 
    puan: Dict[str, float] = defaultdict(float)
    for kanun, kelimeler in _KANUN_KW.items():
        for kelime, agirlik in kelimeler:
            if kelime in q:
                puan[kanun] += agirlik

    if not puan:
        return None

    # Puanları sırala
    sirali = sorted(puan.items(), key=lambda x: -x[1])
    en_iyi, p = sirali[0]

    # --- KRİTİK İYİLEŞTİRME: GÜVEN KONTROLÜ ---
    
    # Kural A: Toplam puan çok düşükse (örn: sadece 1-2 puan), 
    # tesadüfi kelimeler olabilir, kanun filtresi uygulama.
    if p < 4: # Eşik değeri 2'den 4'e çıkarıldı
        log.debug(f"Kanun tespiti yetersiz puan ({p}), genel arama yapılacak.")
        return None

    # Kural B: Belirsizlik Eşiği (Margin of Safety)
    # Eğer en yüksek puanlı iki kanun birbirine çok yakınsa (örn: hem TBK hem TKHK kelimeleri varsa)
    # Yanlış kanunu filtreleyip Recall'u düşürmemek için filtreyi kaldır (None dön).
    if len(sirali) > 1 and sirali[1][1] > 0:
        oran = p / sirali[1][1]
        if oran < 1.8: 
            log.debug(f"Kanun tespiti belirsiz (oran={oran:.2f}), filtre uygulanmıyor.")
            return None

    log.debug(f"Kanun tespiti başarılı: {en_iyi} (puan={p})")
    return en_iyi

def extract_madde(query: str) -> Optional[str]:
    """
    Sorgudan madde numarasını çıkarır.

    Returns:
        "315" | "47" | None
    """
    for m in _MADDE_RE.finditer(query.lower()):
        g = m.groups()
        # Format 1: g[1]=madde
        if g[1] and g[1].isdigit():
            return g[1]
        # Format 2: g[2]=madde (sadece "madde X")
        if g[2] and g[2].isdigit():
            return g[2]
        # Format 3: g[4]=madde
        if g[4] and g[4].isdigit():
            return g[4]
        # Format 4: g[5]=madde
        if g[5] and g[5].isdigit():
            return g[5]
        
    return None


def normalize_article(x: Any) -> str:
    """article_no alanından sadece rakamları çıkarır: "47/1" → "47"."""
    if x is None:
        return ""
    found = re.search(r"(\d+)", str(x))
    return found.group(1) if found else ""


# ═══════════════════════════════════════════════════════════════════════════════
# SORGU GENİŞLETME — BM25 recall artışı için
# ═══════════════════════════════════════════════════════════════════════════════

_ESANLAMLILAR: Dict[str, str] = {
    # Proje formu kapsam terimlerinin eş anlamlıları
    "hata":              "yanılma irade fesadı sakatlık",
    "gabin":             "aşırı yararlanma sömürme oransızlık",
    "temerrüt":          "gecikme ödememe direnim borç ödenmedi",
    "ayıp":              "kusur eksiklik gizli ayıp bozuk",
    "ayıplı ifa":        "eksik ifa hatalı edim kusurlu",
    "fesih":             "sona erdirme bozma iptal feshetme",
    "cayma":             "cayma hakkı dönme geri alma vazgeçme",
    "promosyon":         "hediye kampanya reklam tanıtım",
    "kayıp bedel":       "kayıp kaçak elektrik fatura",
    "erken ödeme":       "peşin vade indirimi önceden ödeme",
    "rekabet yasağı":    "rekabet etmeme yasak ihlali",
    "özen borcu":        "özenle yönetim sadakat dikkat",
    "protesto":          "protesto çekme ret bildirim ihbar",
    "alt kira":          "alt kiracı devir yasak kiralama",
    "iş yeri dışı":      "mesafeli kapıdan ev ziyareti",
    "tahliye":           "boşaltma çıkma kira fesih kiracı",
    "ifa imkânsızlığı":  "imkânsızlık ifa edilememe edim",
    "sebepsiz zenginleşme": "haksız iktisap karşılıksız kazanım",
}


def expand_query(query: str) -> str:
    """Sorguya eş anlamlı terimler ekler → BM25 recall artışı."""
    q    = query.lower()
    ekler = [esanl for anahtar, esanl in _ESANLAMLILAR.items() if anahtar in q]
    if not ekler:
        return query
    genisletilmis = query + " " + " ".join(ekler)
    log.debug(f"Sorgu genişletildi: {len(ekler)} terim eklendi")
    return genisletilmis


# ═══════════════════════════════════════════════════════════════════════════════
# BM25PLUS — Türkçe hukuk metinleri için optimize edilmiş
# ═══════════════════════════════════════════════════════════════════════════════

class BM25Plus:
    """
    BM25+ motoru.

    BM25+'ın BM25'ten farkı: delta parametresi.
    Bu, her terim için minimum katkıyı garanti eder —
    düşük frekanslı ama kritik hukuki terimleri ödüllendirir.

    Parametreler (Türkçe hukuk metinleri için optimize):
        k1=1.6  — terim frekansı doygunluğu
        b=0.68  — belge uzunluğu normalizasyonu
        delta=1.0 — minimum katkı sabiti
    """

    def __init__(self, k1: float = 1.6, b: float = 0.68, delta: float = 1.0):
        self.k1    = k1
        self.b     = b
        self.delta = delta
        self.n     = 0
        self.avgdl = 0.0
        self.idf:  Dict[str, float]     = {}
        self.tf:   List[Dict[str, int]] = []
        self.dl:   List[int]            = []

    def _tokenize(self, text: str) -> List[str]:
        """Basit tokenizasyon — hukuki metinler için yeterli."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 1]

    def index(self, docs: List[str]) -> None:
        """Corpus'u indexler — bir kez çalışır, cache'e kaydedilir."""
        self.n  = len(docs)
        toks    = [self._tokenize(d) for d in docs]
        self.dl = [len(t) for t in toks]
        self.avgdl = sum(self.dl) / max(self.n, 1)

        df: Dict[str, int] = defaultdict(int)
        for t in toks:
            for term in set(t):
                df[term] += 1

        self.idf = {
            term: math.log((self.n - f + 0.5) / (f + 0.5) + 1)
            for term, f in df.items()
        }

        self.tf = []
        for t in toks:
            freq: Dict[str, int] = defaultdict(int)
            for term in t:
                freq[term] += 1
            self.tf.append(dict(freq))

        log.info(f"[BM25+] {self.n} doküman indexlendi.")

    def score(self, query: str, n: int = 300) -> List[Tuple[int, float]]:
        """
        BM25+ puanlaması.
        Sorgu genişletme BM25 içinde uygulanır (dense taraf etkilenmez).
        """
        q_toks = self._tokenize(expand_query(query))
        if not q_toks:
            return []

        scores: List[Tuple[int, float]] = []
        for i in range(self.n):
            dl   = self.dl[i]
            tf   = self.tf[i]
            skor = 0.0
            for t in q_toks:
                if t not in self.idf or tf.get(t, 0) == 0:
                    continue
                f   = tf[t]
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                skor += self.idf[t] * ((num / den) + self.delta)
            if skor > 0:
                scores.append((i, skor))

        scores.sort(key=lambda x: -x[1])
        return scores[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID FUSION
# ═══════════════════════════════════════════════════════════════════════════════

def _minmax_normalize(vals: List[float]) -> List[float]:
    """Min-Max normalizasyon — [0,1] aralığına taşır."""
    if not vals:
        return vals
    mn, mx = min(vals), max(vals)
    rng    = mx - mn
    return [(v - mn) / rng if rng > 0 else 1.0 for v in vals]


def hybrid_fuse(
    dense_hits: List[Dict],
    bm25_hits:  List[Dict],
    alpha:      float,
) -> List[Dict]:
    """
    Dense + BM25 skorlarını normalize edip alpha ile birleştirir.

    Proje formu: α=0.65 (embedding kosinüs + BM25/Jaccard)
    final_score = alpha * dense_norm + (1-alpha) * bm25_norm

    alpha=0.25 → BM25 ağırlıklı (exact match, madde no sorgular)
    alpha=0.65 → Dengeli (varsayılan)
    alpha=0.75 → Dense ağırlıklı (semantik sorgular)
    """
    # Dense normalizasyonu
    d_norms = _minmax_normalize([h["dense_score"] for h in dense_hits])
    dense_map: Dict[str, Dict] = {}
    for h, dn in zip(dense_hits, d_norms):
        dense_map[h["chunk_id"]] = {**h, "dense_norm": dn}

    # BM25 normalizasyonu
    b_norms = _minmax_normalize([h["bm25_score"] for h in bm25_hits])
    bm25_map: Dict[str, Dict] = {}
    for h, bn in zip(bm25_hits, b_norms):
        bm25_map[h["chunk_id"]] = {**h, "bm25_norm": bn}

    # Birleştir — iki listede olan chunk'lar ödüllendirilir
    all_cids = set(dense_map) | set(bm25_map)
    fused: List[Dict] = []

    for cid in all_cids:
        d    = dense_map.get(cid)
        b    = bm25_map.get(cid)
        base = d if d else b

        d_norm = d["dense_norm"] if d else 0.0
        b_norm = b["bm25_norm"]  if b else 0.0

        fused.append({
            **base,
            "skor":        round(alpha * d_norm + (1 - alpha) * b_norm, 6),
            "dense_score": round(d["dense_score"] if d else 0.0, 4),
            "bm25_score":  round(b["bm25_score"]  if b else 0.0, 4),
        })

    fused.sort(key=lambda x: -x["skor"])
    return fused


# ═══════════════════════════════════════════════════════════════════════════════
# LEGAL RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════════

class LegalRetriever:
    """
    TBK/TKHK/TTK kapsamında hibrit hukuki retriever.

    Embedder v10 uyumluluğu:
    - Sorgular encode_single() ile embed edilir — "query: " prefix embedder içinde
    - Corpus "LAW Madde NO: text" formatında embed edildi (embed_corpus)
    - Bu asimetri Mursit modelinin bi-encoder yapısını doğru kullanır

    Shortcut mekanizması:
    - "TBK m.315" sorgusunda m.315 top-100'de yoksa corpus'tan bulup enjekte eder
    - Bu özellikle nadir veya az chunk'lanan maddeler için Recall'u artırır
    """

    def __init__(
        self,
        quantize: bool = False,
        reindex:  bool = False,
        cfg:      CFG  = None,
    ):
        self.cfg      = cfg or CFG()
        self.embedder = MursitEmbedder(quantize=quantize)
        self.qdrant   = QdrantClient(path=QDRANT_STORAGE)
        self.bm25     = BM25Plus()
        self.corpus:  List[Dict] = []

        atexit.register(self._kapat)
        self._load(reindex)

        log.info(
            f"LegalRetriever hazır — {len(self.corpus)} chunk | "
            f"{'int8' if quantize else 'float32'}"
        )

    # ── Yaşam döngüsü ─────────────────────────────────────────────────────────

    def _kapat(self):
        try:
            self.qdrant.close()
        except Exception:
            pass

    # ── Veri yükleme ──────────────────────────────────────────────────────────

    def _load(self, reindex: bool) -> None:
        """
        Corpus'u cache'den veya Qdrant'tan yükler, BM25'i indexler.

        Cache geçerliliği:
        - İlk kez veya reindex=True → Qdrant scroll + BM25 index + cache kaydet
        - Cache varsa → direkt yükle (çok daha hızlı)
        """
        cache_path = self.cfg.CACHE_PATH

        if not reindex and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)

                corpus = data.get("corpus", [])
                bm25   = data.get("bm25")

                if corpus and bm25:
                    self.corpus = corpus
                    self.bm25   = bm25
                    log.info(f"Cache yüklendi: {len(corpus)} chunk ← {cache_path}")
                    return
                else:
                    log.warning("Cache eksik, yeniden oluşturuluyor...")
            except Exception as e:
                log.warning(f"Cache okunamadı ({e}), yeniden oluşturuluyor...")

        # Qdrant'tan scroll ile tüm corpus'u çek
        log.info("Corpus Qdrant'tan yükleniyor (scroll)...")
        corpus: List[Dict] = []
        offset = None

        while True:
            try:
                sonuc, offset = self.qdrant.scroll(
                    COLLECTION_NAME,
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                log.error(f"Qdrant scroll hatası: {e}")
                break

            for p in sonuc:
                pay = p.payload
                corpus.append({
                    "chunk_id":   str(p.id),
                    "text":       pay.get("text", ""),
                    "law":        pay.get("law", ""),
                    "article_no": normalize_article(pay.get("article_no")),
                    "source":     pay.get("source", ""),
                    "decision_id":pay.get("decision_id", ""),
                })

            if offset is None:
                break

        log.info(f"Corpus: {len(corpus)} chunk yüklendi.")

        # BM25 indexle — ham text (prefix olmadan)
        self.bm25.index([c["text"] for c in corpus])
        self.corpus = corpus

        # Cache kaydet
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"corpus": corpus, "bm25": self.bm25}, f)
            size_mb = os.path.getsize(cache_path) / 1024 / 1024
            log.info(f"Cache kaydedildi → {cache_path} ({size_mb:.1f} MB)")
        except Exception as e:
            log.warning(f"Cache kaydedilemedi: {e}")

    # ── Dense arama ───────────────────────────────────────────────────────────

    def _dense(self, query: str, kanun: Optional[str]) -> List[Dict]:
        """
        Qdrant üzerinde dense arama.

        NOT: encode_single() embedder içinde zaten "query: " prefix ekliyor.
        Burada tekrar eklemeye gerek yok.
        """
        vec    = self.embedder.encode_single(query)
        filtre = None
        if kanun:
            filtre = qm.Filter(
                must=[qm.FieldCondition(
                    key="law",
                    match=qm.MatchValue(value=kanun)
                )]
            )

        try:
            sonuc = self.qdrant.query_points(
                COLLECTION_NAME,
                query=vec,
                query_filter=filtre,
                limit=self.cfg.TOP_K_DENSE,
            ).points
        except Exception as e:
            log.error(f"Dense arama hatası: {e}")
            return []

        return [{
            "chunk_id":    str(p.id),
            "text":        p.payload.get("text", ""),
            "law":         p.payload.get("law", ""),
            "article_no":  normalize_article(p.payload.get("article_no")),
            "source":      p.payload.get("source", ""),
            "decision_id": p.payload.get("decision_id", ""),
            "dense_score": round(p.score, 4),
        } for p in sonuc]

    # ── BM25 arama ────────────────────────────────────────────────────────────

    def _sparse(self, query: str, kanun: Optional[str]) -> List[Dict]:
        """
        Gevşetilmiş Sparse Arama:
        Kanun filtresini katı bir 'at' yerine, 'puan kır' mantığına çevirir.
        Böylece farklı kanundaki benzer maddeler aday havuzuna girebilir.
        """
        # Aday havuzunu geniş tutalım (k * 5)
        ham = self.bm25.score(query, n=self.cfg.TOP_K_BM25 * 5)
        sonuclar = []
        
        for idx, skor in ham:
            doc = self.corpus[idx]
            
            # Başlangıç skoru
            final_bm25_skor = skor
            
            # --- GEVŞETME MANTIĞI ---
            if kanun and doc["law"] != kanun:
                # Kanun tutmuyorsa maddeyi çöpe atmıyoruz, 
                # sadece puanını %30 kırarak listenin altına itiyoruz.
                # Bu, 'yanlış kanun tespiti' durumunda Recall'u kurtarır.
                final_bm25_skor = skor * 0.70
            
            sonuclar.append({
                **doc, 
                "bm25_score": round(final_bm25_skor, 4)
            })
            
            # İstenen aday sayısına ulaşınca dur
            if len(sonuclar) >= self.cfg.TOP_K_BM25:
                break
                
        return sonuclar

    # ── Shortcut injection ────────────────────────────────────────────────────

    def _shortcut(
        self,
        fused: List[Dict],
        kanun: Optional[str],
        madde: Optional[str],
    ) -> List[Dict]:
        """
        Madde tespit edildi ama top-100'de yok → corpus'tan manuel enjeksiyon.

        Bu mekanizma nadir/az chunk'lanan maddeler için Recall@10'u artırır.
        Özellikle "TBK m.315" gibi direkt madde sorgularında kritik.
        """
        if not madde or not self.cfg.SHORTCUT_ENABLED:
            return fused

        # Mevcut sonuçlarda bu madde var mı?
        mevcut_maddeler = {normalize_article(c.get("article_no")) for c in fused}
        if madde in mevcut_maddeler:
            return fused  # Zaten var

        log.info(f"Shortcut: {kanun} m.{madde} top-{self.cfg.FINAL_K}'de yok → enjekte ediliyor")

        # Corpus'ta bu maddeyi bul
        adaylar = [
            c for c in self.corpus
            if normalize_article(c.get("article_no")) == madde
            and (kanun is None or c.get("law") == kanun)
        ]

        if not adaylar:
            log.warning(f"Shortcut: {kanun} m.{madde} corpus'ta da bulunamadı!")
            return fused

        # En üste enjekte et
        max_skor = fused[0]["skor"] if fused else 1.0
        enjekte  = adaylar[:self.cfg.SHORTCUT_MAX]

        for aday in enjekte:
            fused.insert(0, {
                **aday,
                "skor":        round(max_skor * 1.2, 6),  # en üste koy
                "dense_score": 0.0,
                "bm25_score":  0.0,
                "shortcut":    True,
            })

        log.info(f"Shortcut: {len(enjekte)} chunk enjekte edildi.")
        return fused

    # ── Madde tekrar filtresi ─────────────────────────────────────────────────

    def _madde_filtrele(self, chunks: List[Dict]) -> List[Dict]:
        """Aynı kanun+madde'den MAX_SAME_ARTICLE chunk al."""
        sayac: Dict[str, int] = defaultdict(int)
        sonuc = []
        for c in chunks:
            art = normalize_article(c.get("article_no"))
            key = f"{c.get('law', '')}_{art}"
            if sayac[key] < self.cfg.MAX_SAME_ARTICLE:
                c["article_no"] = art  # normalize et
                sonuc.append(c)
                sayac[key] += 1
        return sonuc

    # ── Ana retrieval pipeline ────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """
        Hibrit retrieval pipeline (v6.2 - Dinamik Alpha):
        1. Sorgu karakteristiğine göre (uzunluk, madde no) Alpha tayini yapar.
        2. Kanun tespitinde 'Güven Eşiği' kullanarak yanlış filtrelemeyi engeller.
        """
        k = k or self.cfg.FINAL_K

        # 1. Entity extraction (Kanun ve Madde Tespiti)
        kanun = detect_kanun(query)
        madde = extract_madde(query)
        
        # --- DİNAMİK ALPHA VE INTENT TAYİNİ ---
        query_word_count = len(query.split())

        if madde:
            # Eğer sorguda madde no varsa (örn: m.315), anahtar kelime eşleşmesi (BM25) esastır.
            intent = "exact_article"
            alpha = self.cfg.ALPHA_EXACT # Örn: 0.30
        elif query_word_count < 4:
            # Çok kısa sorgularda anlamsal bağlam zayıf olabilir, BM25'i destekle.
            intent = "short_keyword"
            alpha = 0.50 # Dengeli (Dense %50, BM25 %50)
        elif query_word_count > 15:
            # Uzun olay anlatımlarında/senaryolarda anlamsal modele (Mursit) güven.
            intent = "semantic_story"
            alpha = self.cfg.ALPHA_SEMANTIC # Örn: 0.77
        else:
            # Standart sorgular
            intent = "standard_semantic"
            alpha = self.cfg.ALPHA_DEFAULT # Örn: 0.65

        log.info(
            f"Sorgu: '{query[:50]}...' | "
            f"Kanun={kanun or 'Genel'} | Madde={madde or '?'} | "
            f"Intent={intent} | α={alpha}"
        )

        # 2. Dense search (Vektörel Arama)
        dense = self._dense(query, kanun)

        # Fallback: Eğer kanun filtresi varken sonuç çok az gelirse filtresiz arama ile destekle
        if kanun and len(dense) < 10:
            log.debug(f"Dense az sonuç ({len(dense)}), filtresiz fallback ekleniyor...")
            dense_fb  = self._dense(query, kanun=None)
            mevcut_id = {d["chunk_id"] for d in dense}
            for d in dense_fb:
                if d["chunk_id"] not in mevcut_id:
                    dense.append(d)

        # 3. Sparse search (BM25Plus)
        sparse = self._sparse(query, kanun)

        # 4. Hybrid fusion (Dinamik Alpha ile birleştirme)
        fused = hybrid_fuse(dense, sparse, alpha)

        # 5. Boosting (Tam eşleşen maddeleri yukarı taşı)
        for c in fused:
            art = normalize_article(c.get("article_no"))
            if madde and art == madde:
                c["skor"] = round(c["skor"] * self.cfg.BOOST_MADDE, 6)
            if kanun and c.get("law") == kanun:
                c["skor"] = round(c["skor"] * self.cfg.BOOST_KANUN, 6)

        # Minimum skor kontrolü ve yeniden sıralama
        fused = [c for c in fused if c["skor"] >= self.cfg.MIN_SCORE]
        fused.sort(key=lambda x: -x["skor"])

        # 6. Shortcut injection (Bulunamazsa zorla enjekte et)
        fused = self._shortcut(fused, kanun, madde)

        # 7. Madde tekrar filtresi + top-k (Çeşitlilik sağla)
        return self._madde_filtrele(fused)[:k]

    def search(self, query: str, k: int = None) -> List[Dict]:
        """Evaluator uyumluluğu için alias → retrieve()."""
        return self.retrieve(query, k)


# ═══════════════════════════════════════════════════════════════════════════════
# TERMİNAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LawAgent LegalRetriever")
    parser.add_argument("--sorgu",    type=str,  default="Cayma hakkı süresi kaç gündür?")
    parser.add_argument("--k",        type=int,  default=10)
    parser.add_argument("--quantize", action="store_true", help="int8 quantization")
    parser.add_argument("--reindex",  action="store_true", help="Cache yenile")
    parser.add_argument("--debug",    action="store_true", help="DEBUG log")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    r = LegalRetriever(quantize=args.quantize, reindex=args.reindex)

    kanun = detect_kanun(args.sorgu)
    madde = extract_madde(args.sorgu)

    print(f"\n{'═'*70}")
    print(f"  Sorgu : {args.sorgu}")
    print(f"  Kanun : {kanun or 'filtresiz'}")
    print(f"  Madde : {madde or '-'}")
    print(f"{'═'*70}")

    sonuclar = r.retrieve(args.sorgu, k=args.k)

    if not sonuclar:
        print("  Sonuç bulunamadı.")
    else:
        for i, s in enumerate(sonuclar, 1):
            tag = " ★ SHORTCUT" if s.get("shortcut") else ""
            print(
                f"\n  {i:2d}. skor={s['skor']:.4f}"
                f" (dense={s['dense_score']:.3f}, bm25={s['bm25_score']:.3f}){tag}"
            )
            print(f"      {s.get('law','?'):6s}  m.{s.get('article_no','?'):6s}"
                  f"  [{s.get('source','?')}]")
            print(f"      {s.get('text','')[:120]}...")