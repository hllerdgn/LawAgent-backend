import json
import scrapy
from bs4 import BeautifulSoup


class YargitaySpider(scrapy.Spider):
    name = "yargitay"
    allowed_domains = ["karararama.yargitay.gov.tr"]

    start_urls = ["https://karararama.yargitay.gov.tr/"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1.5,
        "COOKIES_ENABLED": True,
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json;charset=UTF-8",
            "Origin": "https://karararama.yargitay.gov.tr",
            "Referer": "https://karararama.yargitay.gov.tr/",
            "X-Requested-With": "XMLHttpRequest",
        },
    }

    # Aranacak kelimeler (projene göre genişlet)
    SEARCH_TERMS = [
        "TBK",
        "TKHK",
        "TTK",
    ]

    PAGE_SIZE = 10
    MAX_PAGE = 1  # test için 1, prod'da yükselt

    def parse(self, response):
        """
        Ana sayfa çağrıldıktan sonra arama başlatılır
        (session/cookie burada oluşur)
        """
        for term in self.SEARCH_TERMS:
            payload = {
                "data": {
                    "aranan": term,
                    "arananKelime": term,
                    "pageSize": self.PAGE_SIZE,
                    "pageNumber": 1,
                }
            }

            yield scrapy.Request(
                url="https://karararama.yargitay.gov.tr/aramalist",
                method="POST",
                body=json.dumps(payload),
                callback=self.parse_search,
                meta={
                    "term": term,
                    "page": 1,
                },
                dont_filter=True,
            )

    def parse_search(self, response):
        """
        Arama sonuçları JSON
        """
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            self.logger.error(
                "JSON decode failed. Response snippet:\n%s",
                response.text[:500],
            )
            return

        results = data.get("data", {}).get("data", [])
        if not results:
            return

        for karar in results:
            karar_id = karar.get("id")
            if not karar_id:
                continue

            url = "https://karararama.yargitay.gov.tr/" f"getDokuman?id={karar_id}"

            yield scrapy.Request(
                url=url,
                callback=self.parse_decision,
                meta={
                    "karar_id": karar_id,
                    "term": response.meta["term"],
                },
                dont_filter=True,
            )

    def parse_decision(self, response):
        """
        Karar HTML → plain text
        """
        try:
            data = json.loads(response.text)
            html = data.get("data", "")
        except Exception:
            return

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if not text:
            return

        yield {
            "decision_id": response.meta["karar_id"],
            "query": response.meta["term"],
            "text": text,
            "source": "yargitay",
        }
