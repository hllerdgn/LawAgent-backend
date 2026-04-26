BOT_NAME = "scraper"

SPIDER_MODULES = [
    "scraper_mevzuat.spiders",
    "scraper_yargi.spiders",
    "scraper_resmigazete.spiders",
]

NEWSPIDER_MODULE = "scraper_mevzuat.spiders"

ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 2

FEED_EXPORT_ENCODING = "utf-8"

ITEM_PIPELINES = {
    "scraper.pipelines.MevzuatPreprocessPipeline": 300,
    "scraper.pipelines.YargiCleanPipeline": 400,
    "scraper.pipelines.ResmiGazetePipeline": 300,
}

# FEEDS = {
#     "data/%(name)s_corpus.json": {
#         "format": "json",
#         "encoding": "utf8",
#         "indent": 2,
#         "overwrite": True,
#     }
# }
#JOBDIR = "crawls/resmigazete"
