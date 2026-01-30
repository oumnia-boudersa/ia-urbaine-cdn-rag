# extractor.py
import time
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)

def prcess_result_text(text: str) -> str:
    """Process and clean the extracted result text."""
    # Remove the "Partager la fiche:" section and social media links
    if "Partager la fiche:" in text:
        text = text.split("Partager la fiche:")[0].strip()
    
    # Remove unwanted UI text elements
    text = text.replace("Envoyer un message", "").strip()
    text = text.replace("Consulter le site Web", "").strip()
    text = text.replace("Signaler une erreur dans cette fiche", "").strip()
    
    # Clean up any multiple consecutive newlines or spaces
    cleaned_text = "\n".join(line for line in text.split("\n") if line.strip())
    return cleaned_text


def get_total_results(driver):
    """Extract total results count from the page."""
    try:
        result_div = driver.find_element(By.CSS_SELECTOR, ".resultats")
        result_text = result_div.text
        logger.info(f"Total results: {result_text}")
        return result_text
    except Exception as e:
        logger.warning(f"Failed to get total results: {e}")
        return None

def get_all_cards_urls(driver):
    """Extract all card URLs with pagination support."""
    all_urls = []
    while True:
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".card-post.card-post-fullwidth"))
            )
            cards = driver.find_elements(By.CSS_SELECTOR, ".card-post.card-post-fullwidth")
            logger.info(f"Cards found on page: {len(cards)}")

            for card in cards:
                href = card.get_attribute("href")
                if href and href not in all_urls:
                    all_urls.append(href)

            # Check for next page
            try:
                next_link = driver.find_element(By.CSS_SELECTOR, "a.next.page-numbers")
                driver.get(next_link.get_attribute("href"))
            except:
                logger.info("No more pages.")
                break
        except Exception as e:
            logger.error(f"Error during URL extraction: {e}")
            break
    logger.info(f"Total URLs collected: {len(all_urls)}")
    return all_urls

def scrape_card_details(driver, url):
    """Scrape a single card details page."""
    wait = WebDriverWait(driver, 10)
    driver.get(url)
    time.sleep(2)  # Allow page to load

    try:
        section = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.section-apropos.active"))
        )
        full_text = section.text.strip()
        full_text = prcess_result_text(full_text)
        # Remove the "Partager la fiche:" section and social media links
        # if "Partager la fiche:" in full_text:
        #     full_text = full_text.split("Partager la fiche:")[0].strip()
        # Remove unwanted UI text elements
        # full_text = full_text.replace("Envoyer un message", "").strip()
        # full_text = full_text.replace("Consulter le site Web", "").strip()
        # full_text = full_text.replace("Signaler une erreur dans cette fiche", "").strip()
        # # Clean up any multiple consecutive newlines or spaces
        # full_text = "\n".join(line for line in full_text.split("\n") if line.strip())

        title = driver.find_element(By.CSS_SELECTOR, "h1.small.coul-blanc").text.strip()
        website_link = None
        for link in driver.find_elements(By.CSS_SELECTOR, "a.bold"):
            if "consulter le site web" in link.text.lower():
                website_link = link.get_attribute("href")
                print(f"Found website link: {website_link}")
                break

        return {
            "Organisation": title,
            "url": url,
            "Site internet": website_link,
            "details": full_text,
            
        }
    except Exception as e:
        logger.warning(f"Failed to scrape details for {url}: {e}")
        return {"url": url, "details": None, "Organisation": None, "Site internet": None}