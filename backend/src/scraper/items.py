import scrapy


class MevzuatItem(scrapy.Item):
    law = scrapy.Field()
    law_no = scrapy.Field()
    article_no = scrapy.Field()
    text = scrapy.Field()
    char_len = scrapy.Field()
    source = scrapy.Field()
    source_url = scrapy.Field()


class YargitayItem(scrapy.Item):
    court = scrapy.Field()
    chamber = scrapy.Field()
    decision_no = scrapy.Field()
    decision_date = scrapy.Field()
    related_law = scrapy.Field()
    text = scrapy.Field()
    char_len = scrapy.Field()
    source = scrapy.Field()
    source_url = scrapy.Field()


class ResmiGazeteItem(scrapy.Item):
    url = scrapy.Field()
    yayin_tarihi = scrapy.Field()
    sayi = scrapy.Field()
    baslik = scrapy.Field()
    ilgili_kanun = scrapy.Field()
    degisiklik_maddeleri = scrapy.Field()
    raw_text = scrapy.Field()
