import scrapy
import re

KANUNLAR = ["6098", "6102", "6502"]


class ResmiGazeteSpider(scrapy.Spider):
    name = "resmigazete"
    allowed_domains = ["resmigazete.gov.tr"]
    start_urls = ["https://www.resmigazete.gov.tr/"]

    def parse(self, response):
        full_text = " ".join(response.xpath("//body//text()").getall())
        match = re.search(r"(\d{5})\s*Sayılı", full_text)
        rg_sayi = match.group(1) if match else None

        # data-modal="True" olan tüm linkleri al
        for link in response.css("a[data-modal='True']"):
            href = link.attrib.get("href", "")
            baslik = link.css("::text").get("").strip()

            if not href or not baslik:
                continue

            if href.endswith(".pdf"):
                continue

            yield response.follow(
                href,
                callback=self.parse_detay,
                meta={"rg_sayi": rg_sayi, "baslik": baslik},
            )

    def parse_detay(self, response):
        full_text = " ".join(response.xpath("//body//text()").getall())
        rg_sayi = response.meta["rg_sayi"]
        baslik = response.meta["baslik"]

        full_text_lower = full_text.lower()
        bulunan_kanunlar = [k for k in KANUNLAR if f"{k} sayılı" in full_text_lower]

        if not bulunan_kanunlar:
            self.logger.debug(f"İlgili kanun yok: {baslik[:50]}")
            return

        madde_pattern = r"MADDE\s*(\d+[A-Za-z/]*)\s*[–\-—]\s*(.*?)(?=MADDE\s*\d+[A-Za-z/]*\s*[–\-—]|\Z)"

        maddeler = re.findall(madde_pattern, full_text, re.DOTALL | re.IGNORECASE)

        self.logger.info(f"✅ Kanun bulundu: {bulunan_kanunlar} → {baslik[:60]}")

        yield {
            "rg_sayi": rg_sayi,
            "url": response.url,
            "baslik": baslik,
            "kanunlar": bulunan_kanunlar,
            "degisen_maddeler": [
                {"madde_no": m[0], "icerik": m[1].strip()} for m in maddeler
            ],
        }
