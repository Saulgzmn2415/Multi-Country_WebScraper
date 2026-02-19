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

                # Wait for main listing container
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='listing'], div[data-clutch], li[class*='provider']"))
                )

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')

                # Try multiple card selectors
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
                        if agency is not None:
                            agencies.append(agency)
                    except Exception as e:
                        logger.warning(f"Failed to parse one card: {e}")
                        continue

                # Debug line
                logger.info(f"Parsed {len(agencies)} agencies so far on page {page} "
                           f"(names found: {sum(1 for a in agencies if a['name'] != 'Unknown Agency')})")

                time.sleep(3.5)  # polite delay

            except Exception as e:
                logger.error(f"Page {page} failed: {e}")
                time.sleep(10)

        return pd.DataFrame(agencies)

    def _parse_clutch_card(self, card, country_code):
        data = {'country': country_code, 'source': 'clutch.co'}

        # ────────────────────────────────────────────────────────
        # NAME - more robust extraction with multiple fallbacks
        # ────────────────────────────────────────────────────────
        name = None

        # 1. Try direct profile link anchor text (most reliable)
        profile_link = card.select_one('a[href*="/profile/"]')
        if profile_link and profile_link.get_text(strip=True):
            name = profile_link.get_text(strip=True)

        # 2. Fallback: look for common heading classes
        if not name:
            for tag in ['h3', 'h2', 'div', 'span']:
                candidates = card.select(f'{tag}[class*="name"], {tag}[class*="title"], {tag}[class*="company"], {tag}[class*="provider"]')
                for c in candidates:
                    text = c.get_text(strip=True)
                    if text and len(text) > 3 and not text.lower().startswith(('visit', 'view', 'contact')):
                        name = text
                        break
                if name:
                    break

        # 3. Last resort: any <a> with reasonable text inside the card
        if not name:
            links = card.select('a')
            for link in links:
                txt = link.get_text(strip=True)
                if txt and len(txt) > 4 and ' ' in txt:  # likely a company name
                    name = txt
                    break

        data['name'] = name.strip() if name else 'Unknown Agency'

       # WEBSITE — robust multi-pattern extraction
        website_tag = None

        # 1. Clutch redirect links (most common)
        patterns = [
            'a[href*="r.clutch.co/redirect"]',
            'a[href*="redirect?url="]',
            'a[href*="redirect?u="]',
            'a[href*="visit-website"]',
            'a[href*="goto"]',
            'a[href^="https://"]:not([href*="/profile/"])'
        ]

        for p in patterns:
            website_tag = card.select_one(p)
            if website_tag:
                break

        raw_website = website_tag['href'] if website_tag else None

        # 2. Clean redirect URLs
        if raw_website and "redirect" in raw_website and "u=" in raw_website:
            try:
                real = raw_website.split("u=")[-1].split("&")[0]
                real = real.replace("%3A", ":").replace("%2F", "/")
                data['website'] = real
            except:
                data['website'] = raw_website
        else:
            data['website'] = raw_website

        # 3. Remove garbage values
        if data['website'] and data['website'].strip() in ["https://", "http://", "https://,"]:
            data['website'] = None


        # ────────────────────────────────────────────────────────
        # LOCATION
        # ────────────────────────────────────────────────────────
        loc_tag = card.select_one('.locality, .location, [class*="location"], .city, span[class*="city"]')
        data['location_city'] = loc_tag.get_text(strip=True) if loc_tag else self.countries[country_code]['name']

        # ────────────────────────────────────────────────────────
        # RATING
        # ────────────────────────────────────────────────────────
        rating_tag = card.select_one('[class*="rating"], .stars, span[class*="number"], .rating__number')
        rating_text = rating_tag.get_text(strip=True) if rating_tag else '0'
        match = re.search(r'(\d+\.?\d*)', rating_text)
        data['clutch_rating'] = float(match.group(1)) if match else 0.0

        # ────────────────────────────────────────────────────────
        # MIN PROJECT SIZE
        # ────────────────────────────────────────────────────────
        min_proj = card.select_one('[data-tooltip*="Minimum"], [class*="budget"], [class*="min-project"], .hourly-rate')
        min_text = min_proj.get_text(strip=True).replace(',', '') if min_proj else '$0'
        match = re.search(r'[\$€£]?(\d+[kKMB]?\+?)', min_text, re.I)
        data['min_project_size_usd'] = self._parse_currency(match.group(1) if match else '0')

        # ────────────────────────────────────────────────────────
        # EMPLOYEES
        # ────────────────────────────────────────────────────────
        emp_tag = card.select_one('span[itemprop="numberOfEmployees"]')
        if not emp_tag:
            emp_tag = card.find(string=lambda t: "Employees" in t or "employees" in t)
        emp_text = (
            emp_tag.get_text(strip=True) 
            if hasattr(emp_tag, "get_text")
            else (emp_tag.strip() if emp_tag else "10-49")
        )    
        data['employees'] = self._parse_employees(emp_text)

        # ────────────────────────────────────────────────────────
        # SERVICES / WP SPECIALIST
        # ────────────────────────────────────────────────────────
        potential_tags = card.select('a, span, div, li', recursive=True)

        services_raw = []
        junk_keywords = ['view profile', 'visit website', 'this provider is available', 'time zone', 'time zones', 'service focus', 'other', '+', 'services provided']

        common_keywords = [
            'web development', 'mobile app development', 'custom software development', 'ai development',
            'ux/ui design', 'web design', 'e-commerce development', 'branding', 'search engine optimization',
            'pay per click', 'digital strategy', 'it staff augmentation', 'low/no code development',
            'blockchain', 'iot development', 'application testing', 'cloud consulting', 'generative ai'
        ]

        for el in potential_tags:
            txt = el.get_text(strip=True)
            if not txt or len(txt) < 5 or len(txt) > 80:
                continue
            txt_lower = txt.lower()
            if any(j in txt_lower for j in junk_keywords):
                continue
            if '%' in txt and any(k in txt_lower for k in common_keywords):
                services_raw.append(txt)
            elif any(k in txt_lower for k in common_keywords) and len(txt.split()) <= 6:
                services_raw.append(txt)

        # Split concatenated items like '35% Web Development65% Mobile App Development'
        services_split = []
        for s in services_raw:
            matches = re.findall(r'(\d+%\s*[^%]+?)(?=\d+%|$)', s)
            if matches:
                services_split.extend(m.strip() for m in matches if m.strip())
            else:
                services_split.append(s.strip())

        # Step 1: Apply the quick name/junk filter
        filtered = [s.strip() for s in services_split if '%' in s]

        # Step 2: Deduplicate (case-insensitive), preserve first occurrence order
        seen = set()
        services_clean = []
        for s in filtered:
            norm = s.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                # Still do basic cleanup (safe & cheap)
                s = s.strip()                     # just in case
                s = s.rstrip(',.')                # remove trailing comma or period if any
                services_clean.append(s)

        # Optional: sort highest percentage first
        def get_percentage(s):
            try:
                return float(s.split('%')[0].strip())
            except:
                return 0

        services_clean.sort(key=get_percentage, reverse=True)

        data['services'] = services_clean

        # WP specialist detection — only from services
        wp_keywords = ['wordpress', 'word press', 'wp development', 'wordpress development']
        data['is_wp_specialist'] = any(
            any(kw in s.lower() for kw in wp_keywords)
            for s in data.get('services', [])
        )

        # Only return if we have at least a name or website
        if data['name'] == 'Unknown Agency' and not data['website']:
            return None

        return data

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