# main.py
import os
import logging
from urllib.parse import urlparse, parse_qs
from scraper import config
from scraper.browser import init_driver
from scraper.extractor import get_total_results, get_all_cards_urls, scrape_card_details
from scraper.pdf_exporter import save_to_pdf

# Setup logging
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from urllib.parse import urlparse, parse_qs

def extract_filter_name(url: str) -> str:
    """
    Extract mc_thematiques value from URL
    Example: mc_thematiques=sante → sante
    """
    query = parse_qs(urlparse(url).query)
    value = query.get("mc_thematiques", ["unknown"])[0]
    return value.replace("-", "_")


def main():
    driver = init_driver(headless=False)
    #all_results = []
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    for site in config.LINKS:
        filter_name = extract_filter_name(site)
        logger.info(f"Scraping filter: {filter_name}")
        print(f"__________ Scraping filter: {filter_name}")
        
        driver.get(site)
        get_total_results(driver)
        
        urls = get_all_cards_urls(driver)
        site_results = []

        for url in urls:
            data = scrape_card_details(driver, url)
            logger.info(f"Scraped: {data['Organisation']}")
            site_results.append(data)

        #all_results.extend(site_results)
        # ---------- SAVE TXT ----------
        txt_path = os.path.join(config.OUTPUT_DIR, f"{filter_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, card in enumerate(site_results, 1):
                f.write(f"===== Organisation {i} =====\n")
                f.write(f"Page URL: {card['url']}\n")
                f.write(f"Website: {card['Site internet']}\n\n")
                f.write(card["details"] or "")
                f.write("\n\n")


    # Save outputs

    # os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # txt_file = os.path.join(config.OUTPUT_DIR, "results.txt")
    # with open(txt_file, "w", encoding="utf-8") as f:
    #     for i, card in enumerate(all_results, 1):
    #         f.write(f"===== Organisation {i} =====\n")
    #         f.write(f"Page URL: {card['url']}\n")
    #         f.write(f"Website: {card['Site internet']}\n\n")
    #         f.write(card["details"] or "")
    #         f.write("\n\n")

    # ---------- SAVE PDF ----------
    pdf_path = os.path.join(config.OUTPUT_DIR, f"{filter_name}.pdf")
    save_to_pdf(site_results, pdf_path)
    logger.info(f"Saved files for {filter_name}")

    driver.quit()

if __name__ == "__main__":
    main()
