from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    source: str  # mevzuat | yargitay | resmigazete
    law: Optional[str]  # TBK | TTK | TKHK
    law_no: Optional[str]  # 6098 | 6102 | 6502
    article_no: Optional[str]  # madde numarası
    chunk_index: int
    chunk_total: int
    token_len: int
    text: str

    # Ek metadata
    rg_sayi: Optional[str] = None  # RG kayıtları için
    decision_id: Optional[str] = None  # Yargitay kararları için
    atiflar: Optional[list] = None  # metin içindeki kanun atıfları
