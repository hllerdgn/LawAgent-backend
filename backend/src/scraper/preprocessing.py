"""
scraper/preprocessing.py
========================
Metin işleme yardımcıları.

Kapsam
------
- Boşluk ve encoding temizliği
- TC kimlik no / kişisel veri maskeleme
- Yargıtay kararları için rol bazlı anonimleştirme
- Token sayımı (tiktoken varsa doğru, yoksa yaklaşık)
- Kanun/madde atıf çıkarımı

Import zinciri: hiçbir proje modülünden import etmez.
"""

import re
import tiktoken

# --------TOKEN SAYICI-------------
try:
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    print("[INFO] tiktoken aktif — doğru token sayımı kullanılıyor.")
except ImportError:

    def count_tokens(text: str) -> int:
        """Metindeki token sayısını kelime bazlı yaklaşık olarak döndürür.
        Türkçe için kelime sayısı x 1.2 ~ token sayısı.
        Kesin sayım için: pip install tiktoken --break-system-packages"""
        return int(len(text.split()) * 1.3)


_KANUN = r"(?:TBK|TTK|TKHK)"
MADDE = r"(?:m\.?|madde)"
NUMARA = r"\d+[A-Za-z]*"
SAYILI = r"\d{4}\s+say[ıi]l[ıi]"

_ATIF_PATTERN = re.compile(
    rf"(?:"
    rf"({SAYILI})"
    rf"|"
    rf"{_KANUN}\s+{MADDE}\s*({NUMARA})"
    rf"|"
    rf"{MADDE}\s+({NUMARA})"
    rf")",
    re.IGNORECASE,
)


def extract_atiflar(text: str) -> list[str]:
    eslesme_gruplari = _ATIF_PATTERN.findall(text)
    # findall birden fazla grup varsa tuple listesi döndürür
    # her tuple'dan boş olmayanları al
    atiflar = []
    for gruplar in eslesme_gruplari:
        for g in gruplar:
            if g:
                atiflar.append(g)
    return list(set(atiflar))


def clean_whitespace(text: str) -> str:
    """Gereksiz boşluk ve satır sonlarını temizler."""
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Mevzuat ve Resmi Gazete metinleri için önişleme.
    Sadece boşluk temizleme ve TC kimlik no maskeleme yapılır.
    Büyük harfli kelime → [KISI] dönüşümü YAPILMAZ.
    """
    text = clean_whitespace(text)

    # TC Kimlik No varsa maskele (sadece açıkça belirtilmişse)
    text = re.sub(
        r"(T\.C\.?\s*Kimlik\s*No|TC\s*Kimlik\s*No)\s*[:\-]?\s*\d+",
        "[ANONIM]",
        text,
        flags=re.IGNORECASE,
    )
    # 11 haneli sayılar TC kimlik no olabilir
    text = re.sub(r"\b\d{11}\b", "[ANONIM]", text)

    return text.strip()


def anonymize_text(text: str) -> str:
    """
    Yargı kararları için anonimleştirme.
    Kişi maskeleme yalnızca DAVACI/DAVALI/SANIK/MÜŞTEKİ
    gibi rol ifadelerinin ardından yapılır.
    Genel metinde kör maskeleme YAPILMAZ.
    """
    # TC Kimlik No
    text = re.sub(r"\b\d{11}\b", "[TC_KIMLIK_NO]", text)

    # Telefon numarası
    text = re.sub(r"\b0?5\d{9}\b", "[TELEFON]", text)

    # E-posta
    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "[EMAIL]",
        text,
    )

    # Kişi adı → SADECE rol ifadesinin ardından maskele
    text = re.sub(
        r"(DAVACI|DAVALI|SANIK|MÜŞTEKİ|VEKİLİ|KATILAN)\s*[:\-]?\s*"
        r"[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]+(?:\s[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]+)*",
        r"\1: [KISI]",
        text,
    )

    return text.strip()
