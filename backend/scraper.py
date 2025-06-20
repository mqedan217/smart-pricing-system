# --- START OF FILE scraper_final.py ---

import asyncio
import random
import requests
import json
import logging
import urllib.parse
import numpy as np
from playwright.async_api import async_playwright, Error as PlaywrightError
from playwright_stealth import stealth
from typing import Optional, Dict, Any, List # For type hints
from motor.motor_asyncio import AsyncIOMotorCollection
# --- Import Configuration ---
try:
    import config # Import the config module
except ImportError:
    print("ERROR: config.py not found. Please ensure it exists.")
    # Define fallback defaults if config cannot be imported
    class ConfigFallback:
        BRIGHTDATA_API_KEY=None # API Key for BrightData
        PROXY_USERNAME=None # Proxy username for BrightData
        PROXY_PASSWORD=None # Proxy password for BrightData
        SCRAPER_USE_PROXY=False # Use proxy for scraping (default: False)
        SCRAPER_PROXY_HOST="res.proxy-sale.com" # Proxy host for Proxy-Sale
        SCRAPER_PROXY_PORTS=list(range(10000,10010)) # Proxy ports for Proxy-Sale
        SCRAPER_USER_AGENTS=[]
        UULE_LOCATIONS={} # UULE locations for Google Shopping
        BRIGHTDATA_API_URL="https://api.brightdata.com/request" # BrightData API URL
        SCRAPER_MAX_PRODUCTS=20 # Max products to scrape per site
        SCRAPER_NAVIGATION_RETRIES=3 # Navigation retries for Playwright
        SCRAPER_NAVIGATION_TIMEOUT_MS=45000 # Navigation timeout for Playwright
        SCRAPER_GOOGLE_MAX_RESULTS=20 # Max results for Google Shopping
        SELECTORS = { # Default empty selectors
             "amazon": {
        "product": "div.s-main-slot div[data-component-type='s-search-result']",
        "title": "h2.a-size-base-plus.a-spacing-none.a-color-base.a-text-normal",
        "price": "span.a-price .a-offscreen",
        "image": "img.s-image",
        "link": "a.a-link-normal.s-no-outline",
        "rating": "span.a-icon-alt",
        "reviews": "span.a-size-base.s-underline-text",
        "shipping": "span.s-align-children-center"
    },
    "ebay": {
        "product": "li.s-item",
        "title": "div.s-item__title",
        "price": "span.s-item__price",
        "image": "div.s-item__image-wrapper img",
        "link": "a.s-item__link",
        "rating": "div.x-star-rating span.clipped",
        "reviews": "span.s-item__reviews-count",
        "shipping": "span.s-item__shipping"
    },
    "opensooq": {
      "product": "a.postListItemData",
      "title": "h2.breakWord.trimTwoLines",
      "price": "div.priceColor",
      "image": "div.postImgCont img",
      "link": "self",
      "shipping": "div.darkGrayColor"
    }
        }
    config = ConfigFallback()
    print("Warning: Using fallback configuration values for scraper.")
# ---------------------------

# --- Logging Setup ---
# Get logger for this module. Assumes root logger is configured elsewhere.
log = logging.getLogger(__name__)
# ---------------------

# --- Constants from Config (Replaces hardcoded definitions) ---
# Secrets/Credentials (read via config, which reads from DB/env)
API_KEY = config.BRIGHTDATA_API_KEY
PROXY_USERNAME = config.PROXY_USERNAME
PROXY_PASSWORD = config.PROXY_PASSWORD

# URLs & Paths
API_URL = config.BRIGHTDATA_API_URL

# Settings
USE_PROXY = config.SCRAPER_USE_PROXY
PROXY_HOST = config.SCRAPER_PROXY_HOST
PROXY_PORTS = config.SCRAPER_PROXY_PORTS # List of ports
USER_AGENTS = config.SCRAPER_USER_AGENTS # List of user agents
UULE_LOCATIONS = config.UULE_LOCATIONS # Dictionary of UULE codes
MAX_PRODUCTS_PER_SITE = config.SCRAPER_MAX_PRODUCTS
NAVIGATION_RETRIES = config.SCRAPER_NAVIGATION_RETRIES
NAVIGATION_TIMEOUT = config.SCRAPER_NAVIGATION_TIMEOUT_MS
GOOGLE_MAX_RESULTS = config.SCRAPER_GOOGLE_MAX_RESULTS

# Selectors (loaded as a dictionary from config)
SELECTORS = config.SELECTORS

# Headers using configured API Key
HEADERS = {}
if API_KEY:
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
else:
    log.error("BrightData API Key not found in configuration. Google Shopping scrape will fail.")

# -------------------------------------------------
# 🔹 Browser Context using config
async def create_browser_context(browser):
    context = None
    page = None
    proxy_settings = None
    try:
        # Use configured proxy settings
        if config.SCRAPER_USE_PROXY:
             if not config.PROXY_USERNAME or not config.PROXY_PASSWORD:
                 log.warning("Proxy enabled in config but username/password missing/empty. Disabling proxy.")
             else:
                selected_port = random.choice(config.SCRAPER_PROXY_PORTS) # Use configured ports
                proxy_settings = {
                    "server": f"http://{config.SCRAPER_PROXY_HOST}:{selected_port}", # Use configured host
                    "username": config.PROXY_USERNAME,
                    "password": config.PROXY_PASSWORD
                }
                log.debug(f"Using proxy: {proxy_settings['server']}")

        # Use configured user agents
        user_agent = random.choice(config.SCRAPER_USER_AGENTS) if config.SCRAPER_USER_AGENTS else "Mozilla/5.0" # Fallback agent
        viewport = {"width": random.randint(1366, 1920), "height": random.randint(768, 1080)} # Keep viewport dynamic

        context = await browser.new_context(
            user_agent=user_agent, proxy=proxy_settings, viewport=viewport,
            java_script_enabled=True, ignore_https_errors=True,
            extra_http_headers={ "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.google.com/" }
        )
        page = await context.new_page()
        await stealth.stealth_async(page)
        log.debug(f"Browser context created (User-Agent: {user_agent[:30]}...)")
        return context, page
    # Keep robust error handling
    except PlaywrightError as e:
        log.error(f"Error creating browser context: {e}", exc_info=True)
        if page: await page.close()
        if context: await context.close()
        return None, None
    except Exception as e:
        log.error(f"Unexpected error creating browser context: {e}", exc_info=True)
        if page: await page.close()
        if context: await context.close()
        return None, None

# 🔹 Random Delay Function (Keep as is)
async def random_delay(min_sec=0.3, max_sec=1.2):
    await asyncio.sleep(random.uniform(min_sec, max_sec))

# 🔹 Retry Mechanism for Page Navigation using config
async def goto_with_retries(page, url, max_retries=config.SCRAPER_NAVIGATION_RETRIES, timeout=config.SCRAPER_NAVIGATION_TIMEOUT_MS):
    """Attempts navigation using retries/timeout from config."""
    for attempt in range(max_retries):
        try:
            log.info(f"Navigating to {url} (Attempt {attempt + 1}/{max_retries})")
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            status = response.status if response else 0
            if not response or status >= 400:
                 log.warning(f"Navigation attempt {attempt + 1} status {status} for {url}")
                 if status >= 400 and status != 404: raise PlaywrightError(f"Page returned status {status}")
            log.info(f"Successfully navigated to {url} (Status: {status})")
            return True
        # Keep robust error handling
        except PlaywrightError as e:
            log.warning(f"Playwright error on attempt {attempt + 1} for {url}: {e}")
            if attempt < max_retries - 1: await random_delay(3, 7)
            else: log.error(f"Navigation failed for {url} after {max_retries} attempts.")
        except Exception as e:
            log.warning(f"Unexpected navigation error attempt {attempt + 1} for {url}: {e}", exc_info=True)
            if attempt < max_retries - 1: await random_delay(3, 7)
            else: log.error(f"Navigation failed for {url} after {max_retries} attempts due to unexpected error.")
    return False

# 🔹 Scroll Down (Keep as is)
async def scroll_down(page, scrolls=2, min_delay=0.5, max_delay=1.5):
    try:
        for i in range(scrolls):
            if page.is_closed(): log.warning("Page closed during scroll."); return
            scroll_amount = int(page.viewport_size['height'] * random.uniform(0.7, 1.5))
            await page.mouse.wheel(0, scroll_amount)
            log.debug(f"Scrolled down ({i+1}/{scrolls})")
            await random_delay(min_delay, max_delay)
    except Exception as e: # Catch generic error during scroll
        log.warning(f"Error during scrolling: {e}", exc_info=False)


# 🔹 Scrape Product Details using config
async def scrape_product(site: str, url: str, selectors: Dict[str, str], browser, max_products: int = config.SCRAPER_MAX_PRODUCTS):
    """Scrapes products using selectors passed as argument (from config), wrapped in error handling."""
    context, page = await create_browser_context(browser) # Uses config internally
    if not context or not page:
        log.error(f"[{site}] Failed to create browser context. Skipping scrape.")
        return None

    results = []
    # Check if essential selectors are present
    if not selectors or not selectors.get("product"):
        log.error(f"[{site}] Cannot scrape: 'product' selector missing in provided selectors dict.")
        return None

    try:
        success = await goto_with_retries(page, url) # Uses config internally
        if not success:
            log.error(f"[{site}] Failed to navigate to URL: {url}")
            return None # Return None on navigation failure

        await random_delay(1, 3)
        await scroll_down(page)
        await random_delay(0.5, 1.5)

        try:
             await page.wait_for_selector(selectors['product'], timeout=15000)
             log.debug(f"[{site}] Product selector '{selectors['product']}' found.")
        except PlaywrightError:
             log.warning(f"[{site}] Timed out waiting for product selector '{selectors['product']}'.")

        product_elements = await page.query_selector_all(selectors['product'])
        log.info(f"[{site}] Found {len(product_elements)} potential elements using selector: {selectors['product']}")

        if not product_elements:
             log.warning(f"[{site}] No elements matched product selector on {url}")
             return []

        # Use max_products from config (passed as argument)
        for rank, product_element in enumerate(product_elements):
            if len(results) >= max_products:
                log.info(f"[{site}] Reached max product limit ({max_products}).")
                break

            title = "N/A" # Default
            try:
                # Use selectors provided in the 'selectors' dict argument
                title_element = await product_element.query_selector(selectors['title']) if selectors.get('title') else None
                title = await title_element.inner_text() if title_element else "N/A"

                price_element = await product_element.query_selector(selectors['price']) if selectors.get('price') else None
                price = await price_element.inner_text() if price_element else "N/A"

                image_element = await product_element.query_selector(selectors['image']) if selectors.get('image') else None
                image_url = await image_element.get_attribute('src') if image_element else "N/A"
                # --- Handle Link ---
                link_selector = selectors.get('link')
                if link_selector == "self":
                    # Container element *is* the link
                    link = await product_element.get_attribute('href')
                elif link_selector:
                    # Find link element within container
                    link_el = await product_element.query_selector(link_selector)
                    link = await link_el.get_attribute('href') if link_el else "N/A"
                else:
                    link = "N/A" # Default if no link selector
                #--- End Link Handling ---

                rating_element = await product_element.query_selector(selectors['rating']) if selectors.get('rating') else None
                rating = await rating_element.inner_text() if rating_element else "N/A"

                review_element = await product_element.query_selector(selectors['reviews']) if selectors.get('reviews') else None
                reviews = await review_element.inner_text() if review_element else "N/A"

                shipping = "N/A"
                if selectors.get('shipping'): # Check if shipping selector exists
                    shipping_element = await product_element.query_selector(selectors['shipping'])
                    shipping = await shipping_element.inner_text() if shipping_element else "N/A"

                # --- Data Cleaning & Site Specific (Keep Original Logic) ---
                title = title.strip() if title else "N/A"
                price = price.strip() if price else "N/A"
                # ... (keep rest of stripping) ...
                image_url = image_url.strip() if image_url else "N/A"
                link = link.strip() if link else "N/A"
                rating = rating.strip() if rating else "N/A"
                reviews = reviews.strip() if reviews else "N/A"
                shipping = shipping.strip() if shipping else "N/A"

                if site == "amazon":
                    if link != "N/A" and not link.startswith(("http:", "https:")):
                        if not link.startswith("https://www.amazon.ae"):
                            link = f"https://www.amazon.ae{link}"
                elif site == "ebay":
                    if "ebaystatic" in image_url: log.debug(f"[{site}] Skipping ebaystatic"); continue

                if title == "N/A" or price == "N/A": log.debug(f"[{site}] Skipping missing title/price"); continue
                # --- End Cleaning ---
                # --- Ensure links are absolute ---
                if link != "N/A" and not link.startswith(("http:", "https://", "www.")): # Check if it's likely relative
                    try:
                        base_url_parsed = urllib.parse.urlparse(url) # 'url' is the search results page URL
                        base_url = f"{base_url_parsed.scheme}://{base_url_parsed.netloc}"
                        absolute_link = urllib.parse.urljoin(base_url, link)
                        log.debug(f"[{site}] Converted relative link '{link}' to absolute: '{absolute_link}'")
                        link = absolute_link # Update the link variable
                    except Exception as url_e:
                        log.warning(f"[{site}] Failed to construct absolute URL for link '{link}': {url_e}")
                        link = "N/A" # Set to N/A if joining fails
                # --- End Ensure links are absolute ---

                results.append({
                    "site": site, "rank": len(results) + 1, "title": title, "price": price,
                    "image_url": image_url, "link": link, "rating": rating, "reviews": reviews,
                    "shipping": shipping
                })

            # Keep robust error handling per element
            except PlaywrightError as el_err: log.warning(f"[{site}] Playwright error on element {rank + 1} ('{title}'): {el_err}")
            except Exception as el_err: log.warning(f"[{site}] Unexpected error on element {rank + 1} ('{title}'): {el_err}", exc_info=False)

        log.info(f"✅ [{site}] Scraped {len(results)} products using config selectors from {url}")
        return results

    # Keep robust error handling for the function
    except PlaywrightError as page_err: log.error(f"[{site}] Playwright error scraping {url}: {page_err}", exc_info=True); return None
    except Exception as e: log.error(f"[{site}] Unexpected error scraping {url}: {e}", exc_info=True); return None
    finally: # Keep robust cleanup
        if page:
            try: await page.close()
            except Exception as pc_err: log.warning(f"[{site}] Error closing page: {pc_err}")
        if context: 
            try: await context.close()
            except Exception as cc_err: log.warning(f"[{site}] Error closing context: {cc_err}")
        log.debug(f"[{site}] Browser context cleanup attempted.")


# 🔹 Amazon Specific Scraper using config
async def scrape_amazon_product(query: str, browser):
    """Scrapes Amazon using selectors from config."""
    url = f"https://www.amazon.ae/s?k={urllib.parse.quote_plus(query)}" # Encode query for URL
    # Get selectors dictionary for 'amazon' from the loaded config
    amazon_selectors = config.SELECTORS.get("amazon", {})
    if not amazon_selectors:
         log.error("Amazon selectors not found in config. Cannot scrape Amazon.")
         return None # Cannot proceed without selectors
    log.info(f"Using Amazon selectors from config: {list(amazon_selectors.keys())}")
    # Pass the loaded selectors and configured max product limit
    return await scrape_product("amazon", url, amazon_selectors, browser, config.SCRAPER_MAX_PRODUCTS)


# 🔹 eBay Specific Scraper using config
async def scrape_ebay_product(query: str, browser):
    """Scrapes eBay using selectors from config."""
    url = f"https://www.ebay.com/sch/i.html?_nkw={urllib.parse.quote_plus(query)}" # Encode query
    # Get selectors dictionary for 'ebay' from the loaded config
    ebay_selectors = config.SELECTORS.get("ebay", {})
    if not ebay_selectors:
         log.error("eBay selectors not found in config. Cannot scrape eBay.")
         return None # Cannot proceed without selectors
    log.info(f"Using eBay selectors from config: {list(ebay_selectors.keys())}")
    # Pass the loaded selectors and configured max product limit
    return await scrape_product("ebay", url, ebay_selectors, browser, config.SCRAPER_MAX_PRODUCTS)

# 🔹 opensooq Specific Scraper using config
async def scrape_opensooq_product(query: str, browser):
    """Scrapes opensooq using selectors from config."""
    url = f"https://jo.opensooq.com/ar/find?search=true&term={urllib.parse.quote_plus(query)}" # Encode query
    # Get selectors dictionary for 'ebay' from the loaded config
    opensooq_selectors = config.SELECTORS.get("opensooq", {})
    if not opensooq_selectors:
         log.error("opensooq selectors not found in config. Cannot scrape opensooq.")
         return None # Cannot proceed without selectors
    log.info(f"Using opensooq selectors from config: {list(opensooq_selectors.keys())}")
    # Pass the loaded selectors and configured max product limit
    return await scrape_product("opensooq", url, opensooq_selectors, browser, config.SCRAPER_MAX_PRODUCTS)


# 🔹 Google Shopping Scraper using config
async def scrape_google_shopping(query: str, country_code: str, max_results: int = config.SCRAPER_GOOGLE_MAX_RESULTS):
    """Fetches Google Shopping via BrightData using config values."""
    # Check if API key is available
    if not config.BRIGHTDATA_API_KEY:
         log.error("[google] BrightData API Key not found in config. Skipping.")
         return None

    # Use UULE locations from config
    uule_param = f"&uule={config.UULE_LOCATIONS.get(country_code.lower(), '')}" if country_code.lower() in config.UULE_LOCATIONS else ""
    encoded_query = urllib.parse.quote_plus(query)
    # Use max_results from config
    search_url = f"https://www.google.com/search?q={encoded_query}&tbm=shop&gl={country_code.lower()}&hl=en&brd_mobile=1{uule_param}&num={min(max_results, 100)}&start=0&brd_json=1"

    payload = { "zone": "serper", "url": search_url, "format": "json", "method": "GET", "country": country_code.upper() }

    log.info(f"[google] Requesting Google Shopping SERP for '{query}' in {country_code.upper()}")
    # Regenerate HEADERS in case API key was loaded late by config
    request_headers = {}
    if config.BRIGHTDATA_API_KEY:
        request_headers = {
            "Authorization": f"Bearer {config.BRIGHTDATA_API_KEY}",
            "Content-Type": "application/json",
        }
    else: # Should have already returned None, but double-check
        log.error("[google] Cannot make request: BrightData API key missing.")
        return None

    try:
        proxies = {"http": None, "https": None} # Keep proxy bypass for requests
        # Use API URL from config
        response = requests.post( config.BRIGHTDATA_API_URL, headers=request_headers, json=payload, timeout=60, proxies=proxies )
        response.raise_for_status()
        data = response.json()

        # Keep robust response processing
        shopping_items = []
        body_data = data.get("body") # Check if body exists first
        if body_data and isinstance(body_data, str):
             print(f"[google] Body data type: {type(body_data)}")
             try: shopping_items = json.loads(body_data).get("shopping", [])
             except json.JSONDecodeError: log.error("[google] Failed to decode 'body' string.")
        elif isinstance(data.get("shopping"), list): shopping_items = data["shopping"]
        else: log.warning("[google] 'shopping' results key not found or invalid.")

        if not shopping_items: log.warning(f"[google] No shopping items found for '{query}'."); return []

        # Keep result extraction logic
        results = []
        for item in shopping_items[:max_results]: # Apply max_results limit
             results.append({
            "site": "google shopping-" + item.get("shop", ""),  # ✅ Extract store name
            "rank": item.get("rank", len(results)+1),  # ✅ Extract product rank
            "title": item.get("title", "N/A"),
            "price": item.get("price", "N/A"),
            "image": item.get("image_url", "N/A"),
            "link": item.get("shop_link") or item.get("product_link") or item.get("offers_link") or item.get("link", "N/A"), # ✅ Extract product link
            "shipping": item.get("shipping", "N/A"),  # ✅ Extract shipping details
            "old_price": item.get("old_price", "N/A"),  # ✅ Extract discount price
            "rating": item.get("rating", "N/A"),  # ✅ Extract rating
            "reviews": item.get("reviews_cnt", "N/A"),  # ✅ Extract number of reviews
        })

        log.info(f"✅ [google] Processed {len(results)} Google Shopping results.")
        return results

    # Keep robust error handling
    except requests.exceptions.HTTPError as http_err: log.error(f"[google] HTTP error: {http_err}", exc_info=True); return None
    except requests.exceptions.RequestException as e: log.error(f"[google] Request failed: {e}", exc_info=True); return None
    except Exception as e: log.error(f"[google] Unexpected error processing response: {e}", exc_info=True); return None

# --- Feature Storage using config ---
# Pass the collection object from orchestrator for better control
async def store_input_features(featuresVector, productkeyword, category, features_coll : AsyncIOMotorCollection):
    """Store input features. Uses passed collection object."""
    # If collection wasn't passed, try the global one (initialized via config)
    if features_coll is None:
            log.error("MongoDB collection (features_collection) was not provided to store_input_features.")
            return False

    if featuresVector is None:
        log.warning(f"Cannot store features for '{productkeyword}': None vector.")
        return False

    try:
        # Keep numpy conversion/validation
        if isinstance(featuresVector, np.ndarray):
            if featuresVector.ndim > 1: featuresVector = featuresVector.flatten()
            featuresVector = featuresVector.astype('float32')
        else: featuresVector = np.array(featuresVector, dtype='float32').flatten()
        if featuresVector.ndim != 1: log.error(f"Vector not 1D for '{productkeyword}'."); return False

        update_result = await features_coll.update_one(
            {"_id": category},
            {"$addToSet": { "products": { "title": productkeyword, "image_url": "user input", "vector": featuresVector.tolist() }}},
            upsert=True
        )
        # Keep improved logging/return based on result
        if update_result.upserted_id: log.info(f"Stored features (new cat): '{productkeyword}'."); return True
        elif update_result.modified_count > 0: log.info(f"Stored features (new prod): '{productkeyword}'."); return True
        elif update_result.matched_count > 0: log.info(f"Features likely exist for '{productkeyword}'."); return False
        else: log.warning(f"DB Store weird result for '{productkeyword}'."); return False
    # Keep robust error handling
    except Exception as e: log.error(f"Unexpected store error for '{productkeyword}': {e}", exc_info=True); return False


# --- Run All Scraping using config ---
async def scrape_all(query: str, country_code: str = "jo"): # Default country from UULE list example
    """Runs all scrapers concurrently using config values."""
    log.info(f"Starting scrape_all for query: '{query}' in country: {country_code}")
    results = { "amazon_products": None, "ebay_products": None, "google_shopping": None, "opensooq_products": None }
    browser = None

    # Keep async context manager for Playwright
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch( headless=True, args=["--disable-blink-features=AutomationControlled"] )
            log.info("Playwright browser launched.")

            # Prepare tasks
            tasks = [
                scrape_amazon_product(query, browser), # Uses config internally
                scrape_ebay_product(query, browser),   # Uses config internally
                scrape_google_shopping(query, country_code), # Uses config internally
                scrape_opensooq_product(query, browser) # Uses config internally
            ]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results (Keep robust handling)
            site_keys = ["amazon_products", "ebay_products", "google_shopping", "opensooq_products"]
            for i, result in enumerate(task_results):
                key = site_keys[i]
                if isinstance(result, Exception): log.error(f"Scraper {key} failed: {result}", exc_info=result); results[key] = None
                else:
                    results[key] = result
                    if results[key] is None: log.warning(f"Scraper {key} returned None.")
                    elif isinstance(results[key], list): log.info(f"Scraper {key} returned {len(results[key])} results.")
                    else: log.warning(f"Scraper {key} returned unexpected type: {type(results[key])}"); results[key] = None
        # Keep robust error handling for Playwright startup
        except PlaywrightError as pw_err: log.error(f"Playwright error during scrape_all: {pw_err}", exc_info=True)
        except Exception as e: log.error(f"Unexpected error during scrape_all: {e}", exc_info=True)
        # Browser closing handled by context manager

    log.info(f"scrape_all finished for query: '{query}'.")
    # Ensure lists are returned
    results["amazon_products"] = results.get("amazon_products", []) or []
    results["ebay_products"] = results.get("ebay_products", []) or []
    results["google_shopping"] = results.get("google_shopping", []) or []
    results["opensooq_products"] = results.get("opensooq_products", []) or []
    return results
# --- END OF FILE scraper_final.py ---