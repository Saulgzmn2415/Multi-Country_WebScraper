""" Multi-Country Web Agency Scraper
Targets: Clutch.co (primarily), can extend to GoodFirms
Focus: Web development / WordPress / related agencies in SA, UK, US, AU
"""

import re
import time
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalPartnerScraper:
    def __init__(self):
        self.countries = {
            'US': {'name': 'United States', 'slug': 'us', 'currency': 'USD'},
            'UK': {'name': 'United Kingdom', 'slug': 'uk', 'currency': 'GBP'},
            'AU': {'name': 'Australia', 'slug': 'au', 'currency': 'AUD'},
            'SA': {'name': 'South Africa', 'slug': 'za', 'currency': 'ZAR'},
        }
        self.setup_driver()

    def setup_driver(self):
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
        self.driver = webdriver.Chrome(options=options)

    def scrape_clutch_agencies(self, country_code, category='web-developers', max_pages=5):
        if country_code not in self.countries:
            raise ValueError(f"Unknown country: {country_code}")

        slug = self.countries[country_code]['slug']
        base_url = f"https://clutch.co/{slug}/{category}" if slug else f"https://clutch.co/{category}"

        agencies = []
        for page in range(1, max_pages + 1):
            try:
                url = f"{base_url}?page={page}" if page > 1 else base_url
                logger.info(f"Scraping {country_code} - {category} - Page {page}: {url}")
                self.driver.get(url)
                time.sleep(4)  # Give React time to hydrate

                # Wait for main listing container (more reliable than old 'provider')
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='listing'], div[data-clutch], li[class*='provider']"))
                )

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')

                # Modern card selectors — try multiple patterns
                card_selectors = [
                    'div.sponsor-listing',           # sponsored
                    'div.provider-row',              # organic
                    'div.w-100 div.relative',        # common wrapper
                    'li[class*="listing"]',          # list items
                ]

                agency_cards = []
                for sel in card_selectors:
                    found = soup.select(sel)
                    if found:
                        agency_cards = found
                        logger.info(f"Found {len(agency_cards)} cards using selector: {sel}")
                        break

                if not agency_cards:
                    logger.warning("No agency cards found — page structure likely changed.")
                    continue

                for card in agency_cards:
                    try:
                        agency = self._parse_clutch_card(card, country_code)
                        if agency:
                            agencies.append(agency)
                    except Exception as e:
                        logger.warning(f"Failed to parse one card: {e}")
                        continue

                time.sleep(3.5)  # polite delay

            except Exception as e:
                logger.error(f"Page {page} failed: {e}")
                time.sleep(10)

        return pd.DataFrame(agencies)

    def _parse_clutch_card(self, card, country_code):
        data = {'country': country_code, 'source': 'clutch.co'}

        # Name
        name_tag = card.select_one('h3, .company-name, .title, a[href*="/profile/"], span.company')
        data['name'] = name_tag.get_text(strip=True) if name_tag else 'Unknown'

        # Website
        website_tag = card.select_one('a.website-link, a[href^="http"]:not([href*="/profile"]), a[rel="nofollow"]')
        data['website'] = website_tag['href'] if website_tag else None

        # Location (city or detail)
        loc_tag = card.select_one('.locality, .location, span[class*="location"], .city')
        data['location_city'] = loc_tag.get_text(strip=True) if loc_tag else self.countries[country_code]['name']

        # Rating
        rating_tag = card.select_one('.rating, span.rating, [class*="rating__number"], .stars')
        rating_text = rating_tag.get_text(strip=True) if rating_tag else '0'
        data['clutch_rating'] = float(re.search(r'[\d.]+', rating_text).group()) if re.search(r'[\d.]+', rating_text) else 0.0

        # Min project size
        min_proj = card.select_one('[data-tooltip*="Minimum"], .min-project-size, [class*="budget"], .hourly-rate')
        min_text = min_proj.get_text(strip=True).replace(',', '') if min_proj else '$0'
        match = re.search(r'[\$€£]?(\d+[kK]?\+?)', min_text)
        data['min_project_size_usd'] = self._parse_currency(match.group(1) if match else '0')

        # Employees
        emp_tag = card.select_one('.employees, [class*="team"], .size, span[class*="employees"]')
        emp_text = emp_tag.get_text(strip=True) if emp_tag else '10-49'
        data['employees'] = self._parse_employees(emp_text)

        # Services (tags)
        service_tags = card.select('a.tag, .service, [class*="service-tag"], .tag-cloud a')
        data['services'] = [t.get_text(strip=True) for t in service_tags]
        data['is_wp_specialist'] = any('wordpress' in s.lower() for s in data['services'])

        return data if data['name'] != 'Unknown' else None

    @staticmethod
    def _parse_currency(text):
        text = text.replace(',', '').replace('+', '').upper()
        match = re.search(r'(\d+(?:\.\d+)?)[KMB]?', text)
        if not match:
            return 0
        val = float(match.group(1))
        if 'K' in text:
            val *= 1000
        elif 'M' in text:
            val *= 1_000_000
        return int(val)

    @staticmethod
    def _parse_employees(text):
        nums = re.findall(r'\d+', text)
        if len(nums) >= 2:
            return (int(nums[0]) + int(nums[1])) // 2
        elif nums:
            return int(nums[0])
        return 25

    def scrape_all_countries(self, category='web-developers', max_pages=10):
        all_data = []
        for code in self.countries:
            logger.info(f"\n{'='*60}\nSCRAPING {code} - {self.countries[code]['name']}\n{'='*60}")
            df = self.scrape_clutch_agencies(code, category=category, max_pages=max_pages)
            all_data.append(df)
            logger.info(f"{code}: {len(df)} agencies")
            time.sleep(6)

        combined = pd.concat(all_data, ignore_index=True)
        combined.drop_duplicates(subset=['name', 'website'], keep='first', inplace=True)
        combined.to_csv('data/raw/global_partners_raw.csv', index=False)
        logger.info(f"\nTOTAL: {len(combined)} unique agencies")
        return combined

    def __del__(self):
        try:
            self.driver.quit()
        except:
            pass


if __name__ == "__main__":
    scraper = GlobalPartnerScraper()
    try:
        df = scraper.scrape_all_countries(category='web-developers', max_pages=3)  # start small!
        print(df.head(8))
        print("\nCountry breakdown:\n", df['country'].value_counts())
    finally:
        del scraper  # ensure quit