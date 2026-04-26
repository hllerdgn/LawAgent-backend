from scrapy.exceptions import DropItem
from scraper.preprocessing import preprocess_text, anonymize_text, clean_whitespace
import json
import hashlib
from datetime import datetime
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))

CORPUS_FILE = os.path.join(DATA_PATH, "mevzuat_corpus.json")
RG_CACHE_FILE = os.path.join(DATA_PATH, "rg_cache.json")
RG_CORPUS_FILE = os.path.join(DATA_PATH, "resmigazete_corpus.json")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preprocess_rg_text(text: str) -> str:
    """
    Resmi Gazete metinlerine özel önişleme.
    Kişi adı maskeleme YAPILMAZ, sadece temizlik ve TC kimlik no maskeleme.
    """
    # Boşluk ve satır sonu temizle
    text = clean_whitespace(text)
    # Madde başı parantez numaralarını düzenle: ( 1 ) → (1)
    text = re.sub(r"\(\s*(\d+)\s*\)", r"(\1)", text)
    # Virgül ve noktalı virgül sonrası boşluk ekle
    text = re.sub(r"([;,])(?!\s)", r"\1 ", text)
    # TC kimlik no maskele (kişi adı maskeleme olmadan)
    text = preprocess_text(text)
    return text.strip()


class MevzuatPreprocessPipeline:
    def process_item(self, item, spider):
        if spider.name != "mevzuat":
            return item
        temiz = preprocess_text(item["text"])
        if len(temiz) < 50:
            raise DropItem("Metin çok kısa")
        item["text"] = temiz
        item["char_len"] = len(temiz)
        return item


class YargiCleanPipeline:
    MIN_CHAR_LEN = 300

    def process_item(self, item, spider):
        if spider.name != "yargitay":
            return item
        text = item.get("text")
        if not text:
            raise DropItem("Metin alanı boş")
        # Önce temizle, sonra bağlam bazlı anonimleştir
        temiz = preprocess_text(text)
        temiz = anonymize_text(temiz)
        if len(temiz) < self.MIN_CHAR_LEN:
            raise DropItem("Karar metni çok kısa")
        item["text"] = temiz
        item["char_len"] = len(temiz)
        item.setdefault("source", "yargitay")
        return item


class ResmiGazetePipeline:

    def process_item(self, item, spider):
        if spider.name != "resmigazete":
            return item

        rg_sayi = item["rg_sayi"]

        if not self.rg_changed(rg_sayi):
            spider.logger.info("RG aynı → STOP")
            return item

        if not item["kanunlar"]:
            spider.logger.info("İlgili kanun yok → STOP")
            self.update_cache(rg_sayi)
            return item

        item = self.preprocess_item(item)
        self.save_to_corpus(item)
        self.check_changes(item)
        self.update_cache(rg_sayi)
        return item

    def preprocess_item(self, item):
        item["baslik"] = preprocess_rg_text(item.get("baslik", ""))

        temiz_maddeler = []
        for madde in item.get("degisen_maddeler", []):
            temiz_icerik = preprocess_rg_text(madde["icerik"])
            temiz_maddeler.append(
                {
                    "madde_no": madde["madde_no"],
                    "icerik": temiz_icerik,
                    "karakter_sayisi": len(temiz_icerik),
                }
            )

        item["degisen_maddeler"] = temiz_maddeler
        return item

    def save_to_corpus(self, item):
        os.makedirs(DATA_PATH, exist_ok=True)

        mevcut = []
        if os.path.exists(RG_CORPUS_FILE):
            with open(RG_CORPUS_FILE, "r", encoding="utf-8") as f:
                try:
                    mevcut = json.load(f)
                except json.JSONDecodeError:
                    mevcut = []

        guncellendi = False
        for i, kayit in enumerate(mevcut):
            if kayit.get("rg_sayi") == item["rg_sayi"]:
                mevcut[i] = item
                guncellendi = True
                break

        if not guncellendi:
            mevcut.append(item)

        with open(RG_CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(mevcut, f, ensure_ascii=False, indent=2)

    def rg_changed(self, rg_sayi):
        if not os.path.exists(RG_CACHE_FILE):
            return True
        with open(RG_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache.get("last_rg_sayi") != rg_sayi

    def update_cache(self, rg_sayi):
        os.makedirs(DATA_PATH, exist_ok=True)
        with open(RG_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_rg_sayi": rg_sayi,
                    "last_checked_at": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def check_changes(self, item):
        if not os.path.exists(CORPUS_FILE):
            return
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        degisen_maddeler = item.get("degisen_maddeler", [])
        if not degisen_maddeler:
            return

        updated = False
        for madde in degisen_maddeler:
            madde_no = madde["madde_no"]
            yeni_metin = madde["icerik"]

            for kanun in item["kanunlar"]:
                key = f"{kanun}_{madde_no}"
                if key not in corpus:
                    continue

                current = corpus[key]["versions"][-1]
                yeni_hash = sha(yeni_metin)

                if yeni_hash == current["hash"]:
                    continue

                corpus[key]["versions"].append(
                    {
                        "version": current["version"] + 1,
                        "rg_sayi": item["rg_sayi"],
                        "rg_tarih": datetime.now().date().isoformat(),
                        "text": yeni_metin,
                        "hash": yeni_hash,
                    }
                )
                corpus[key]["current_version"] += 1
                self.re_embed(key, yeni_metin)
                updated = True

        if updated:
            with open(CORPUS_FILE, "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False, indent=2)

    def re_embed(self, key, text):
        print(f"{key} yeniden embed edildi")
