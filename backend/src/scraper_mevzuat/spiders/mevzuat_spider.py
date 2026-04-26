import re
import scrapy
from scraper.items import MevzuatItem


class MevzuatSpider(scrapy.Spider):
    name = "mevzuat"
    allowed_domains = ["mevzuat.gov.tr"]

    custom_settings = {
        "DOWNLOAD_DELAY": 2,
        "ROBOTSTXT_OBEY": True,
    }

    # Kanunlar
    kanunlar = {
        "TBK": "6098",
        "TTK": "6102",
        "TKHK": "6502",
    }

    def start_requests(self):
        for law, law_no in self.kanunlar.items():
            url = f"https://www.mevzuat.gov.tr/MevzuatMetin/1.5.{law_no}.htm"
            yield scrapy.Request(
                url=url,
                callback=self.parse_kanun,
                meta={
                    "law": law,
                    "law_no": law_no,
                },
            )

    def parse_kanun(self, response):
        law = response.meta["law"]
        law_no = response.meta["law_no"]

        # Tüm body text
        raw_text = response.xpath("//body//text()").getall()
        raw_text = "\n".join(t.strip() for t in raw_text if t.strip())

        # MADDE bazlı ayırma
        pattern = r"(MADDE\s+\d+)"
        parts = re.split(pattern, raw_text)

        for i in range(1, len(parts), 2):
            article_no = parts[i].replace("MADDE", "").strip()
            article_text = parts[i + 1].strip()

            if len(article_text) < 50:
                continue

            yield MevzuatItem(
                law=law,
                law_no=law_no,
                article_no=article_no,
                text=article_text,
                char_len=len(article_text),
                source="mevzuat",
                source_url=response.url,
            )
