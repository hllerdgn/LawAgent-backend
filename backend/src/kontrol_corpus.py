"""
kontrol_corpus.py — Ground truth maddelerinin corpus'ta olup olmadığını kontrol eder
"""

import json

with open("data/chunk_corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

# Kontrol edilecek maddeler — evaluator.py'deki ground truth
KONTROL = {
    "TBK": [
        "315",
        "316",
        "317",  # kira temerrüdü
        "30",
        "31",
        "32",  # sözleşme iptali
        "434",
        "435",  # hizmet feshi
        "120",
        "121",  # temerrüt faizi
        "581",
        "582",
        "583",
    ],  # kefalet
    "TKHK": [
        "47",
        "48",
        "49",  # cayma hakkı
        "11",
        "12",  # ayıplı mal
        "56",
        "57",  # garanti
        "66",
        "67",
        "68",
    ],  # tüketici hakem
    "TTK": [
        "335",
        "336",
        "337",  # anonim kuruluş
        "573",
        "574",  # limited sorumluluk
        "790",
        "791",
        "792",  # çek karşılıksız
        "11",
        "12",  # ticari devir
        "407",
        "408",
        "409",
    ],  # genel kurul
}

print(f"\n{'═'*60}")
print(f"  CORPUS MADDE KONTROL")
print(f"  Toplam chunk: {len(corpus)}")
print(f"{'═'*60}")

toplam_eksik = 0

for kanun, maddeler in KONTROL.items():
    print(f"\n  {kanun}")
    print(f"  {'─'*50}")
    for madde in maddeler:
        eslesme = [
            c for c in corpus if c.get("law") == kanun and c.get("article_no") == madde
        ]
        if eslesme:
            print(
                f"    m.{madde:<6} ✓  {len(eslesme)} chunk — {eslesme[0]['text'][:60]}..."
            )
        else:
            print(f"    m.{madde:<6} ✗  CORPUS'TA YOK")
            toplam_eksik += 1

print(f"\n{'═'*60}")
print(f"  Eksik madde sayısı: {toplam_eksik}")
print(f"{'═'*60}\n")
