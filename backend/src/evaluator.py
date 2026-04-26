"""
evaluator.py — LawAgent Retrieval Değerlendirme Modülü
=======================================================
Proje: TÜBİTAK 2209/A — RAG Tabanlı Akıllı Hukuk Asistanı

Proje formu Hedef 1 metrikleri:
  Recall@10  ≥ 0.75  — doğru kaynağın ilk 10'da bulunma oranı
  MRR@10     ≥ 0.60  — ilk doğru kaynağın sıralamadaki konumu
  nDCG@10    ≥ 0.70  — doğru kaynakların sıralama kalitesi
  Hit Rate@10         — en az 1 doğru sonuç bulunan sorgu oranı

Ek metrikler:
  Precision@k         — ilk k'daki doğru sonuç oranı

Benchmark: Proje formu kapsamı (TBK + TKHK) — 30 sorgu
  - TBK: satım, hizmet, eser, kira, temerrüt, ayıplı ifa, geçersizlik
  - TKHK: ayıplı mal, cayma hakkı, garanti, mesafeli sözleşme, tüketici

Kullanım:
  python evaluator.py                      # float32, k=10, 30 sorgu
  python evaluator.py --quantize           # int8
  python evaluator.py --k 5               # k=5
  python evaluator.py --rapor             # detaylı sorgu bazlı rapor
  python evaluator.py --analiz            # başarısız sorgular kök neden
  python evaluator.py --json              # JSON çıktısı
  python evaluator.py --debug             # DEBUG log
"""

import argparse
import json
import math
import os
import time
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

from retriever import LegalRetriever, detect_kanun, extract_madde

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LawAgent.Evaluator")
 
BENCHMARK: List[Tuple[str, str, List[str]]] = [
# TKHK - Finansal Hizmetler ve Krediler
    ("Kredi kartı yapılandırmasında akdi faiz oranının üst sınırı nasıl belirlenir?", "TKHK", ["26"]),
    ("Tüketici kredisinde sigorta yaptırılmaması durumunda bankanın kredi şartlarını ağırlaştırma yetkisi var mıdır?", "TKHK", ["29"]),
    ("Kredi sözleşmesinden cayma hakkı kullanan tüketicinin ödediği ücretlerin iade süresi nedir?", "TKHK", ["30"]),
    ("Konut finansmanında erken ödeme tazminatı hangi oranla sınırlandırılmıştır?", "TKHK", ["37"]),
    ("Değişken faizli konut kredisinde endeks değişikliği tüketiciye ne kadar süre önce bildirilmelidir?", "TKHK", ["33"]),
    
    # TKHK - Ayıplı Mal ve Hizmet
    ("Ayıplı malın neden olduğu ölüm veya yaralanma durumunda üreticinin tazminat sorumluluğu nedir?", "TKHK", ["11", "12"]),
    ("İkinci el ürün satışında ayıp sorumluluğu için kararlaştırılabilecek asgari süre nedir?", "TKHK", ["12"]),
    ("Ayıplı hizmette bedel indirimi hakkı hangi durumlarda tüketicinin yararına olur?", "TKHK", ["15"]),
    ("Garanti belgesi düzenleme zorunluluğu hangi tür mallar için geçerlidir?", "TKHK", ["56"]),
    ("İthal edilen ürünlerde satış sonrası servis istasyonu kurma zorunluluğu süresi nedir?", "TKHK", ["58"]),

    # TKHK - Pazarlama ve Sözleşme Türleri
    ("Süreli yayın promosyonlarında 'kültürel ürün' kapsamına girmeyen hediyelerin yasal durumu nedir?", "TKHK", ["21"]),
    ("İnternet üzerinden alınan dijital içeriklerde cayma hakkı hangi aşamada sona erer?", "TKHK", ["48"]),
    ("Kapıdan satışlarda ön bilgilendirme yükümlülüğü ihlal edilirse cayma süresi ne kadar uzar?", "TKHK", ["47"]),
    ("Devre tatil sözleşmelerinde tüketicinin kişisel veri paylaşımına zorlanması haksız şart mıdır?", "TKHK", ["5", "50"]),
    ("Paket turda mücbir sebep nedeniyle iptal durumunda organizatörün iade yükümlülüğü nedir?", "TKHK", ["51"]),
    # TBK - Genel Hükümler
    ("Sözleşmede edimler arasındaki aşırı oransızlık (gabin) durumunda iptal davası açma süresi ne zaman başlar?", "TBK", ["28"]),
    ("Korkutma (ikrah) etkisiyle yapılan sözleşmenin onaylanmış sayılması için gereken şartlar nelerdir?", "TBK", ["39"]),
    ("Temsilcinin yetki sınırlarını aşarak yaptığı işlemlerde temsil olunanın sorumluluğu nedir?", "TBK", ["46", "47"]),
    ("Müteselsil borçlulardan birinin borcu ikrar etmesi zamanaşımını diğerleri için keser mi?", "TBK", ["155", "167"]),
    ("Borca aykırılık durumunda cezai şartın hakim tarafından re'sen indirilmesi mümkün müdür?", "TBK", ["182"]),

    # TBK - Kira Hukuku
    ("Konut kiralarında kiracının özen borcuna aykırılık nedeniyle sözleşmenin feshi süreci nasıl işler?", "TBK", ["316"]),
    ("Kira bedelinin belirlenmesinde TÜFE oranının üzerinde artış yapılmasının hukuki geçerliliği nedir?", "TBK", ["344"]),
    ("Alt kiracının, asıl kiracının kira süresinden daha uzun bir süre için hak sahibi olması mümkün müdür?", "TBK", ["322"]),
    ("Kiralananın aile konutu olması durumunda fesih bildiriminin eşe de yapılması zorunlu mu?", "TBK", ["349"]),
    ("Kira sözleşmesinde yazılı olmayan yan giderlerin ödenmemesi tahliye nedeni midir?", "TBK", ["314", "315"]),

    # TBK - Özel Borç İlişkileri
    ("Hizmet sözleşmesinde işçinin rekabet yasağı kaydının geçerli olması için gereken şartlar nelerdir?", "TBK", ["444", "445"]),
    ("Eser sözleşmesinde müteahhidin işi zamanında bitiremeyeceğinin anlaşılması durumunda iş sahibinin dönme hakkı?", "TBK", ["473"]),
    ("Vekalet sözleşmesinde vekilin alt vekil atama yetkisi hangi durumlarda doğar?", "TBK", ["507"]),
    ("Kefalette eşin rızasının aranmadığı istisnai ticari haller nelerdir?", "TBK", ["584"]),
    ("Sebepsiz zenginleşmede zenginleşenin elinde kalan miktarın tespiti nasıl yapılır?", "TBK", ["79"]),
   # TTK - Ticari İşletme ve Tacir
    ("Bir esnafın işletme kapasitesini aşarak ticari faaliyet yürütmesi durumunda tacir sayılma kriterleri nedir?", "TTK", ["11", "15"]),
    ("Ticaret unvanına tecavüz halinde açılacak davalarda zamanaşımı süresi nedir?", "TTK", ["52"]),
    ("Haksız rekabet teşkil eden beyanların yayılması durumunda içeriğin kaldırılması davası kimlere karşı açılır?", "TTK", ["56"]),
    ("Ticari defterlerin usulüne uygun tutulmaması halinde tacire uygulanacak idari yaptırımlar nelerdir?", "TTK", ["64"]),
    ("Acentenin müvekkil adına sözleşme yapma yetkisinin tescil ve ilan edilmesinin etkisi nedir?", "TTK", ["107"]),

    # TTK - Şirketler Hukuku
    ("Anonim şirket yönetim kurulu üyelerinin rekabet yasağına aykırı davranmasının tazminat sorumluluğu?", "TTK", ["396"]),
    ("Limited şirket müdürlerinin şirkete verdikleri zarardan dolayı sorumluluk davası açma yetkisi kimdedir?", "TTK", ["644", "553"]),
    ("Şirket genel kurul kararlarının emredici hükümlere aykırı olması durumunda butlan davası süresi?", "TTK", ["447"]),
    ("Anonim şirketlerde rüçhan hakkının kısıtlanması için genel kurulda gereken karar yeter sayısı nedir?", "TTK", ["461"]),
    ("Hamiline yazılı pay senetlerinin MKK'ya bildirilmemesinin oy hakkı üzerindeki etkisi nedir?", "TTK", ["489"]),

    # TTK - Kıymetli Evrak
    ("Çekin arkasına vurulan 'karşılıksızdır' şerhinin hukuki niteliği ve ibraz süresi ilişkisi nedir?", "TTK", ["796", "814"]),
    ("Poliçede kabul etmeme protestosu çekilmemesinin müracaat hakları üzerindeki etkisi nedir?", "TTK", ["713", "730"]),
    ("Zamanaşımına uğramış bononun temel ilişkiye dayalı alacak davasında ispat gücü nedir?", "TTK", ["732"]),]
# ═══════════════════════════════════════════════════════════════════════════════
# METRİK FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════════════════

def recall_at_k(results: List[Dict], expected: List[str], k: int) -> float:
    """
    Recall@k — Beklenen maddelerin kaçı ilk k'da bulundu?

    Formül: |bulunan ∩ beklenen| / |beklenen|

    Örnek:
        beklenen = ["315", "316"]  → 2 madde
        ilk 10'da sadece "315" var → 1 bulundu
        Recall@10 = 1/2 = 0.50
    """
    if not expected:
        return 0.0
    found = {r["article_no"] for r in results[:k]}
    return len(found & set(expected)) / len(expected)


def precision_at_k(results: List[Dict], expected: List[str], k: int) -> float:
    """
    Precision@k — İlk k sonucun kaçı doğru?

    Formül: |bulunan ∩ beklenen| / k
    """
    if not results:
        return 0.0
    found = {r["article_no"] for r in results[:k]}
    return len(found & set(expected)) / k


def mrr_at_k(results: List[Dict], expected: List[str], k: int) -> float:
    """
    MRR@k — Mean Reciprocal Rank (tek sorgu için).

    İlk doğru sonucun sırası neyse 1/sıra döndürür.

    Örnek:
        1. sıra: m.999 → yanlış
        2. sıra: m.315 → DOĞRU → 1/2 = 0.500
        MRR = 0.500
    """
    for rank, r in enumerate(results[:k], 1):
        if r["article_no"] in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: List[Dict], expected: List[str], k: int) -> float:
    """
    nDCG@k — Normalized Discounted Cumulative Gain.

    MRR'dan farkı: birden fazla doğru sonucun sıralamasını da dikkate alır.

    Formül:
        DCG  = Σ rel_i / log2(i+1)   (i=1..k, rel_i=1 doğruysa, 0 yanlışsa)
        IDCG = ideal DCG (tüm doğrular üstte olsaydı)
        nDCG = DCG / IDCG

    Örnek:
        Beklenen: ["315", "316"]
        1.sıra: m.316 (doğru), 2.sıra: m.999 (yanlış), 3.sıra: m.315 (doğru)

        DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.000 + 0 + 0.500 = 1.500
        IDCG = 1/log2(2) + 1/log2(3) = 1.000 + 0.631 = 1.631
        nDCG = 1.500 / 1.631 = 0.920
    """
    if not expected:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, r in enumerate(results[:k], 1)
        if r["article_no"] in expected
    )
    n_ilgili = min(len(expected), k)
    idcg     = sum(1.0 / math.log2(i + 1) for i in range(1, n_ilgili + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(results: List[Dict], expected: List[str], k: int) -> int:
    """Hit@k — En az 1 doğru sonuç var mı? (1 veya 0)"""
    found = {r["article_no"] for r in results[:k]}
    return 1 if found & set(expected) else 0


# ── Birim testler ──────────────────────────────────────────────────────────────

def _test_metrikler() -> None:
    """Metriklerin doğruluğunu doğrular — her çalıştırmada otomatik."""
    dummy    = [
        {"article_no": "315"}, {"article_no": "999"},
        {"article_no": "316"}, {"article_no": "888"}, {"article_no": "317"},
    ]
    expected = ["315", "316", "317"]

    # Recall: 3/3 ilk 5'te
    assert abs(recall_at_k(dummy, expected, 5) - 1.0) < 1e-9, "Recall@5 hatası"
    # Recall: ilk 2'de sadece 315 → 1/3
    assert abs(recall_at_k(dummy, expected, 2) - 1/3) < 1e-9, "Recall@2 hatası"
    # MRR: 315 ilk sırada → 1.0
    assert abs(mrr_at_k(dummy, expected, 5) - 1.0) < 1e-9, "MRR hatası"
    # MRR: boş sonuç → 0.0
    assert mrr_at_k([], expected, 5) == 0.0, "MRR boş hatası"
    # Precision@5: 3 doğru / 5 = 0.6
    assert abs(precision_at_k(dummy, expected, 5) - 3/5) < 1e-9, "Precision hatası"
    # nDCG: geçerli aralıkta
    ndcg_val = ndcg_at_k(dummy, expected, 5)
    assert 0.0 < ndcg_val <= 2.0, f"nDCG aralık hatası: {ndcg_val}"
    # Hit: var
    assert hit_rate_at_k(dummy, expected, 5) == 1, "Hit hatası"
    # Hit: yok
    assert hit_rate_at_k([{"article_no":"999"}], expected, 1) == 0, "Hit negatif hatası"

    log.info("[Birim Testler] ✓ Tüm 8 metrik testi geçti.")


# ═══════════════════════════════════════════════════════════════════════════════
# DEĞERLENDİRİCİ
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    retriever:  LegalRetriever,
    benchmark:  List[Tuple[str, str, List[str]]],
    k:          int  = 10,
    verbose:    bool = True,
) -> Dict:
    """
    Tüm benchmark sorgularını çalıştırır, metrikleri hesaplar.

    Returns:
        {
            "recall": float, "mrr": float, "ndcg": float,
            "precision": float, "hit_rate": float,
            "n_sorgu": int, "detay": List[dict]
        }
    """
    recall_l, mrr_l, ndcg_l, prec_l, hit_l = [], [], [], [], []
    detay_l = []

    for sorgu, kanun, beklenen in benchmark:
        t0      = time.time()
        results = retriever.search(sorgu, k=k)
        sure_ms = round((time.time() - t0) * 1000)

        rec  = recall_at_k(results, beklenen, k)
        mrr  = mrr_at_k(results, beklenen, k)
        ndcg = ndcg_at_k(results, beklenen, k)
        prec = precision_at_k(results, beklenen, k)
        hit  = hit_rate_at_k(results, beklenen, k)

        recall_l.append(rec)
        mrr_l.append(mrr)
        ndcg_l.append(ndcg)
        prec_l.append(prec)
        hit_l.append(hit)

        getirilen   = {r["article_no"] for r in results[:k]}
        bulunan     = sorted(getirilen & set(beklenen))
        bulunamayan = sorted(set(beklenen) - getirilen)

        detay = {
            "sorgu":          sorgu,
            "kanun":          kanun,
            "beklenen":       beklenen,
            "retrieved":      [str(r.get("article_no")) for r in results[:k]],
            "recall":         round(rec,  4),
            "mrr":            round(mrr,  4),
            "ndcg":           round(ndcg, 4),
            "precision":      round(prec, 4),
            "hit":            hit,
            "sure_ms":        sure_ms,
            "bulunan":        bulunan,
            "bulunamayan":    bulunamayan,
            "tespit_kanun":   detect_kanun(sorgu) or "—",
            "tespit_madde":   extract_madde(sorgu) or "—",
            "sonuclar": [
                {
                    "rank":       i + 1,
                    "law":        r.get("law", ""),
                    "article_no": r.get("article_no", ""),
                    "skor":       round(r.get("skor", 0.0), 4),
                    "hit":        r.get("article_no", "") in beklenen,
                    "shortcut":   r.get("shortcut", False),
                }
                for i, r in enumerate(results[:k])
            ],
        }
        detay_l.append(detay)

        if verbose:
            flag = "✓" if mrr > 0 else "✗"
            scut = " [S]" if any(r.get("shortcut") for r in results[:k]) else ""
            print(
                f"  {flag} [{kanun:4s}] {sorgu[:48]:<48} "
                f"Rec={rec:.3f}  MRR={mrr:.3f}  nDCG={ndcg:.3f}  ({sure_ms}ms){scut}"
            )

    n = len(benchmark)
    return {
        "recall":    round(sum(recall_l) / n, 4),
        "mrr":       round(sum(mrr_l)    / n, 4),
        "ndcg":      round(sum(ndcg_l)   / n, 4),
        "precision": round(sum(prec_l)   / n, 4),
        "hit_rate":  round(sum(hit_l)    / n, 4),
        "n_sorgu":   n,
        "detay":     detay_l,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RAPORLAMA
# ═══════════════════════════════════════════════════════════════════════════════

_HEDEFLER = {
    "recall":    (0.75, "Recall@k"),
    "mrr":       (0.60, "MRR@k"),
    "ndcg":      (0.70, "nDCG@k"),
}


def _yazdir_ozet(sonuc: Dict, k: int, label: str) -> None:
    """Proje formu hedeflerine göre özet tablo."""
    print(f"\n{'█'*70}")
    print(f"  LawAgent Retrieval Değerlendirme — {label} (k={k})")
    print(f"  {sonuc['n_sorgu']} sorgu  |  TBK + TKHK benchmark seti")
    print(f"{'█'*70}")
    print(f"\n  {'Metrik':<18} {'Sonuç':>8} {'Hedef':>8} {'Fark':>8} {'Durum':>10}")
    print(f"  {'─'*55}")

    for key, (hedef, etiket) in _HEDEFLER.items():
        deger = sonuc[key]
        fark  = deger - hedef
        durum = "✅ GEÇTİ" if deger >= hedef else "❌ EKSİK"
        fark_str = f"+{fark:.3f}" if fark >= 0 else f"{fark:.3f}"
        metrik_adi = etiket.replace("k", str(k))
        print(f"  {metrik_adi:<18} {deger:>8.4f} {hedef:>8.2f} {fark_str:>8} {durum:>10}")

    print(f"  {'Precision@'+str(k):<18} {sonuc['precision']:>8.4f} {'—':>8}")
    hit_n = int(sonuc["hit_rate"] * sonuc["n_sorgu"])
    print(
        f"\n  Hit Rate@{k}  : {sonuc['hit_rate']:.4f}"
        f"  ({hit_n}/{sonuc['n_sorgu']} sorguda en az 1 doğru madde)"
    )
    print(f"{'█'*70}\n")


def _yazdir_detay(sonuc: Dict, k: int) -> None:
    """Sorgu bazlı detaylı performans tablosu."""
    print(f"\n{'═'*80}")
    print(f"  SORGU BAZLI DETAY (k={k})")
    print(f"{'═'*80}")
    print(f"  {'#':<3} {'K':<5} {'Rec':>5} {'MRR':>5} {'nDCG':>5} {'Prec':>5}"
          f"  {'TK':>3} {'TM':>4}  Sorgu")
    print(f"  {'─'*75}")

    for i, d in enumerate(sonuc["detay"], 1):
        flag = "✓" if d["hit"] else "✗"
        scut = "S" if any(r.get("shortcut") for r in d["sonuclar"]) else " "
        print(
            f"  {i:<3} {d['kanun']:<5} {d['recall']:>5.3f} {d['mrr']:>5.3f} "
            f"{d['ndcg']:>5.3f} {d['precision']:>5.3f}  "
            f"{d['tespit_kanun'][:3]:>3} {d['tespit_madde'][:4]:>4}"
            f"  {flag}{scut} {d['sorgu'][:40]}"
        )

    # Kanun bazlı özet
    print(f"\n  {'─'*75}")
    print(f"  KANUN BAZLI ORTALAMA:")
    for kanun in ["TKHK", "TBK", "TTK"]:
        grup = [d for d in sonuc["detay"] if d["kanun"] == kanun]
        if not grup:
            continue
        n   = len(grup)
        ort = lambda key: sum(d[key] for d in grup) / n
        hit = sum(d["hit"] for d in grup)
        print(
            f"  {kanun:<5} ({n} sorgu)  "
            f"Recall={ort('recall'):.3f}  MRR={ort('mrr'):.3f}  "
            f"nDCG={ort('ndcg'):.3f}  Hit={hit}/{n}"
        )


def _yazdir_analiz(sonuc: Dict, k: int) -> None:
    """Başarısız ve düşük performanslı sorguların kök neden analizi."""
    print(f"\n{'═'*80}")
    print(f"  KÖK NEDEN ANALİZİ — Hedef Altı Sorgular")
    print(f"{'═'*80}")

    # Sıfır hit
    sifir = [d for d in sonuc["detay"] if d["hit"] == 0]
    if sifir:
        print(f"\n  [KRİTİK] Hiç doğru madde bulunamayan: {len(sifir)} sorgu")
        for d in sifir:
            print(f"    • {d['sorgu'][:60]}")
            print(f"      Beklenen: {d['beklenen']}")
            print(f"      Tespit: kanun={d['tespit_kanun']}, madde={d['tespit_madde']}")
            print(f"      İlk 3 sonuç: {[r['law']+' m.'+r['article_no'] for r in d['sonuclar'][:3]]}")

    # Düşük Recall
    dusuk_rec = [d for d in sonuc["detay"] if 0 < d["recall"] < 0.75]
    if dusuk_rec:
        print(f"\n  [DÜŞÜK RECALL] Recall < 0.75: {len(dusuk_rec)} sorgu")
        for d in sorted(dusuk_rec, key=lambda x: x["recall"]):
            print(f"    • Recall={d['recall']:.3f} | {d['sorgu'][:50]}")
            print(f"      Bulunamayan: {d['bulunamayan']}")

    # Düşük MRR (doğru bulundu ama üst sırada değil)
    dusuk_mrr = [d for d in sonuc["detay"] if 0 < d["mrr"] < 0.5]
    if dusuk_mrr:
        print(f"\n  [DÜŞÜK MRR] İlk sırada değil: {len(dusuk_mrr)} sorgu")
        for d in sorted(dusuk_mrr, key=lambda x: x["mrr"]):
            rank = round(1 / d["mrr"]) if d["mrr"] > 0 else "?"
            print(f"    • MRR={d['mrr']:.3f} (≈{rank}. sırada) | {d['sorgu'][:50]}")

    # Kanun tespiti başarısızlıkları
    tespit_fail = [d for d in sonuc["detay"] if d["tespit_kanun"] == "—"]
    if tespit_fail:
        print(f"\n  [TESPİT] Kanun tespit edilemeyen: {len(tespit_fail)} sorgu")
        for d in tespit_fail:
            print(f"    • [{d['kanun']}] {d['sorgu'][:60]}")

    if not sifir and not dusuk_rec and not dusuk_mrr:
        print("  ✅ Tüm sorgular hedef metriklerin üzerinde.")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ANA PROGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Birim testler — her çalıştırmada otomatik
    _test_metrikler()

    parser = argparse.ArgumentParser(
        description="LawAgent Retrieval Evaluator — TÜBİTAK 2209/A"
    )
    parser.add_argument("--quantize", action="store_true",  help="int8 model")
    parser.add_argument("--reindex",  action="store_true",  help="Cache yenile")
    parser.add_argument("--k",        type=int, default=10, help="Top-k (varsayılan: 10)")
    parser.add_argument("--rapor",    action="store_true",  help="Sorgu bazlı detay")
    parser.add_argument("--analiz",   action="store_true",  help="Kök neden analizi")
    parser.add_argument("--json",     action="store_true",  help="JSON çıktısı")
    parser.add_argument("--debug",    action="store_true",  help="DEBUG log")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    fmt   = "int8" if args.quantize else "float32"
    label = f"Mursit/{fmt}"

    print(f"\n[Evaluator] Başlatılıyor...")
    print(f"  Model  : {label}")
    print(f"  k      : {args.k}")
    print(f"  Sorgu  : {len(BENCHMARK)}")
    print(f"  Hedefler: Recall≥0.75 | MRR≥0.60 | nDCG≥0.70")

    retriever = LegalRetriever(quantize=args.quantize, reindex=args.reindex)

    print(f"\n[Evaluator] Benchmark sorguları çalıştırılıyor...\n")
    sonuc = evaluate(retriever, BENCHMARK, k=args.k, verbose=True)

    _yazdir_ozet(sonuc, args.k, label)

    if args.rapor:
        _yazdir_detay(sonuc, args.k)

    if args.analiz:
        _yazdir_analiz(sonuc, args.k)

    if args.json:
        cikti = {
            "tarih":    datetime.now().isoformat(),
            "model":    label,
            "k":        args.k,
            "n_sorgu":  sonuc["n_sorgu"],
            "recall":   sonuc["recall"],
            "mrr":      sonuc["mrr"],
            "ndcg":     sonuc["ndcg"],
            "precision":sonuc["precision"],
            "hit_rate": sonuc["hit_rate"],
            "hedefler": {
                "recall_hedef": 0.75,
                "mrr_hedef":    0.60,
                "ndcg_hedef":   0.70,
            },
            "detay": sonuc["detay"],
        }
        os.makedirs("data", exist_ok=True)
        zaman = datetime.now().strftime("%Y%m%d_%H%M")
        yol   = f"data/eval_{fmt}_k{args.k}_{zaman}.json"
        with open(yol, "w", encoding="utf-8") as f:
            json.dump(cikti, f, ensure_ascii=False, indent=2)
        print(f"  JSON kaydedildi → {yol}")


if __name__ == "__main__":
    main()