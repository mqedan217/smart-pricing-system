# --- START OF FILE timegpt.py ---
from currency_converter import CurrencyConverter
from nixtla.date_features import CountryHolidays
from datetime import datetime, timedelta
from nixtla import NixtlaClient
import pandas as pd
import random
import asyncio
import random
import urllib.parse
from playwright.async_api import async_playwright
from playwright_stealth import stealth
import config 
import logging
import httpx
from playwright_stealth import stealth_async 
import re
# Settings
USE_PROXY = config.SCRAPER_USE_PROXY
PROXY_USERNAME = config.PROXY_USERNAME
PROXY_PASSWORD = config.PROXY_PASSWORD
PROXY_HOST = config.SCRAPER_PROXY_HOST
PROXY_PORTS = config.SCRAPER_PROXY_PORTS # List of ports
USER_AGENTS = config.SCRAPER_USER_AGENTS # List of user agents
PROXY_SERVER = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:"
NIXTLA_API_KEY = config.NIXTLA_API_KEY
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

# Define the payload for shopHistory
PRICESPY_URL = "https://pricespy.co.uk/_internal/bff" # PriceSpy API endpoint
PRICESPY_SHOP_IDS = [29044, 1751] # eBay and Amazon(1751) Marketplace shop IDs
PRICESPY_TIME_RANGE = "All"
##########################################################################################
# --- Logging Setup ---
# Get logger for this module. Assumes root logger is configured elsewhere.
log = logging.getLogger(__name__)
# ---------------------
# Initialize CurrencyConverter
try:
    c = CurrencyConverter()
    # Consider pre-loading common currencies if possible?
except Exception as e:
    log.error(f"Failed to initialize CurrencyConverter: {e}")
    c = None


# === Step 1: Finding PriceSpy Item ID ===
async def _create_browser_context_for_id(browser):
    proxy_settings = None
    if USE_PROXY:
        proxy_settings = {"server": f"http://{PROXY_HOST}:{random.choice(PROXY_PORTS)}","username": PROXY_USERNAME,
            "password": PROXY_PASSWORD} 
        log.debug("Using proxy for ID search")
    else:
        log.debug("Not using proxy for ID finding.")
    
    context = await browser.new_context(
        user_agent=random.choice(USER_AGENTS),
        proxy=proxy_settings,
        viewport={"width": random.randint(1280, 1920), "height": random.randint(720, 1080)},
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9"},
    )

    page = await context.new_page()    # This creates a new tab in the browser context.
    await stealth_async(page)  # Apply stealth mode to avoid bot detection
    return context, page
###############################################################
async def _goto_with_retries_async(page, url, max_retries=3, timeout=30000):
    for attempt in range(max_retries):
        try:
            log.info(f"🔎 Navigating to {url} (Attempt {attempt + 1}/{max_retries})")
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            log.info(f"Navigation successful for {url[:60]}...")
            return True
        except Exception as e:
            log.warning(f"⚠️ Attempt {attempt + 1} failed for {url}: {e}")
            await asyncio.sleep(random.uniform(2, 5))  # Longer wait on failure
    log.error(f"❌ All navigation attempts failed for {url}")
    return False
########################################################################
async def _get_first_google_result(query, site):
    """
    Get only the first Google search result for a specific site
    """
    google_url = f"https://www.google.com/search?q=site:{site}+{query}"
    browser = None # Initialize to None
    context = None # Initialize to None
    try: 
        async with async_playwright() as p:        #This ensures Playwright is properly closed when the function finishes.
            browser = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            context, page = await _create_browser_context_for_id(browser)
            try:
                success = await _goto_with_retries_async(page, google_url)
                if not success:
                    return None
                log.debug(f"Querying all 'a' elements on {google_url}") 
                elements = await page.query_selector_all("a")
                log.debug(f"Found {len(elements)} 'a' elements.")
                for element in elements:
                    href = await  element.get_attribute("href")
                    if href and site in href:
                        clean_url = href.split("&")[0]
                        if clean_url.startswith("/url?q="):
                            clean_url= urllib.parse.unquote(clean_url.replace('/url?q=', ""))
                        if site in clean_url and "translate.google.com" not in clean_url:
                            log.info(f"✅ Found first valid result (Original Logic): {clean_url}")
                            return clean_url
                log.warning(f"❌ No valid result found matching criteria for query '{query}' after checking {len(elements)} links.")
                return None
            except Exception as e:
                log.error(f"Playwright Error during Google search processing for '{query}': {e}", exc_info=True)
            finally:
                if context: await context.close()
                if browser: await browser.close()
                log.debug("Playwright browser/context closed.")
    except Exception as setup_e : 
        log.error(f"Playwright setup error for '{query}': {setup_e}", exc_info=True)
        if browser and not browser.is_close():
            await browser.close()
            return None
##################################################################################
def _extract_item_id(url) :
    """Internal helper to extract item ID from PriceSpy URL."""
    if not url: 
        return None
    try:
        p_index = url.rfind('--p')
        if p_index != -1:
            item_id_str = url[p_index+3 : ]
            try:
                return int(item_id_str)
            except ValueError:
                log.error(f"Extracted ID '{item_id_str}' is not integer from URL: {url}")
                return None
        else:
            match = re.search(r'/product\.php\?p=(\d+)', url)
            if match:
                item_id_str = match.group(1)
                try:
                    return int(item_id_str)
                except ValueError:
                    log.error(f"Extracted ID '{item_id_str}' is not integer from URL: {url}")
                    return None
            else:
                return None
    except Exception as e:
        log.error(f"Error extracting ID from {url}: {e}", exc_info=True)
        return None
    ###################################################
async def find_pricespy_id(item_name):
    """
    Finds the PriceSpy item ID for a given item name using Google Search.
    Returns the integer ID or None if not found or an error occurs.
    NOTE: This function uses Playwright and should be run in a background task.
    """
    log.info(f"Starting PriceSpy ID search for item: '{item_name}'")
    site = "pricespy.co.uk"
    encoded_query = urllib.parse.quote_plus(item_name)

    # Call internal helper that uses Playwright
    first_result_url = await _get_first_google_result(encoded_query, site)

    if not first_result_url:
        log.warning(f"ID search failed: No PriceSpy URL found on Google for '{item_name}'.")
        return None

    # Call internal helper to extract ID from the found URL
    item_id = _extract_item_id(first_result_url)

    if item_id:
        log.info(f"ID search successful: Found ID {item_id} for '{item_name}' from URL {first_result_url}")
    else:
        log.warning(f"ID search failed: Could not extract valid ID from URL {first_result_url} for '{item_name}'.")
    return item_id

# ==================================================================
# === Step 2: Fetching Historical Data (Async HTTPX)            ===
# ==================================================================######## Now extracting the historical data for the ID we extracted ###################################

async def fetch_historical_data_async(item_id):
    """Fetches historical price data from the PriceSpy API asynchronously."""
    if not isinstance(item_id, int) or item_id <= 0:
        log.error(f"Invalid item_id provided: {item_id}")
        return None

    log.info(f"Fetching historical data for item ID: {item_id}")
    payload = {
    "operationName": "shopHistory",
    "query": """
        query shopHistory($id: Int!, $timeRange: TimeRange!, $shopIds: [Int!]!) {
            product(id: $id) {
                shopHistory(timeRange: $timeRange, shopIds: $shopIds) {
                    shopId
                    productHistory {
                        historyItems {
                            date
                            name
                            price
                            isMembershipPrice
                            isPaidMembership
                            shopName
                            shopId
                            active
                        }
                        oldestHistoryItem
                        newestHistoryItem
                    }
                }
            }
        }
    """,
    "variables": {
        "id": item_id,
        "timeRange": PRICESPY_TIME_RANGE,
        "shopIds": PRICESPY_SHOP_IDS
    }
}
 
    # Use httpx.AsyncClient for connection pooling
    async with httpx.AsyncClient(timeout=20.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(PRICESPY_URL, headers=headers, json=payload)
                response.raise_for_status()
                log.info(f"API call successful for item {item_id} (Status: {response.status_code})")
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = random.uniform(5, 15) * (attempt + 1) 
                    log.warning(f"Rate limited (429) for item {item_id}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                elif e.response.status_code >= 500:
                     wait_time = random.uniform(3, 7) * (attempt + 1) 
                     log.warning(f"Server error ({e.response.status_code}) for item {item_id}. Retrying in {wait_time:.1f}s...")
                     await asyncio.sleep(wait_time)
                else:
                    log.error(f"HTTP Client Error fetching data for item {item_id}: {e}")
                    return None 
            except httpx.RequestError as e:
                 wait_time = random.uniform(3, 7) * (attempt + 1) 
                 log.warning(f"Network/Request Error fetching data for item {item_id}: {e}. Retrying in {wait_time:.1f}s...")
                 await asyncio.sleep(wait_time)
            except Exception as e:
                 log.error(f"Unexpected error fetching data for item {item_id}: {e}", exc_info=True)
                 return None

    log.error(f"Failed to fetch data for item {item_id} after multiple attempts.")
    return None

# ==================================================================
# === Step 3: Processing Data (Pandas - Synchronous)            ===
# ==================================================================
def process_data_to_df(data):
    """Processes raw API response JSON into a Pandas DataFrame."""
    if not data:
        return None
    
    shop_history = data.get("data", {}).get("product", {}).get("shopHistory", [])
    extracted_data = []
    
    for shop in shop_history:
        history_items = shop.get("productHistory", {}).get("historyItems", [])
        for item in history_items:
            extracted_data.append({
                "shopName": item.get("shopName"),
                "ds": item.get("date"),
                "price": item.get("price"),
                "item": item.get("name")
            })

    if not extracted_data: 
        log.warning("No valid history items found after processing.")
        return None
    df = pd.DataFrame(extracted_data)
    df['ds'] = pd.to_datetime(df['ds']).dt.date
    log.info(f"Processed data into DataFrame")
    df_ebay = df[df["shopName"] == "eBay"]
    df_amazon = df[df["shopName"]=="Amazon.co.uk"]
    # mainly we are going to use amazon data since its not fluctuating as ebay data but if there is no amazon data we will use ebay data 
    if df_amazon is None:
        return df_ebay
    return df_amazon

# === Step 4: Filling Gaps ===
"""
This step ensures that all missing dates between two known timestamps are filled in the dataset. 
For the dates between two known changes,
the price from the previous change is carried forward (according to priceSpy price trend graph).
When a new price change occurs, the subsequent dates will be filled with the updated price.

"""
# Start from the first row and iterate through pairs of consecutive rows
def gaps_filler(df):
    if df is None or df.empty or 'ds' not in df.columns or 'price' not in df.columns:
        log.warning("Cannot fill gaps: DataFrame is invalid or empty.")
        return df # Return original or None
    log.info("Starting gap filling process...")
    df['ds'] = pd.to_datetime(df['ds']) # Ensure 'ds' is datetime
    if len(df) <= 1:
        log.info("Not enough data points to fill gaps.")
        return df # No gaps to fill if 0 or 1 point
    new_rows = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        # Add the current row with its price
        new_rows.append({'ds': current_row ['ds'], 'price': current_row['price']})

        # If there is a gap between dates, fill it with the current price
        if (next_row['ds'] - current_row['ds']).days > 1:
            current_date = current_row['ds']
            while current_date < next_row['ds'] :
                current_date += timedelta(days=1)
                if(current_date ==next_row['ds']  ):
                    continue
                new_rows.append({'ds': current_date, 'price': current_row['price']})
    # Add the last row with its price 
    new_rows.append({'ds': df.iloc[-1]['ds'], 'price': df.iloc[-1]['price']})
    # Create a new DataFrame from new_rows
    return pd.DataFrame(new_rows)

# ==================================================================
# === Step 5: Currency Conversion  ===
# ==================================================================
# Function to find the closest available date with exchange rate data
def get_closest_date(target_date, max_days_back=30):
    
    # Find the closest date with valid exchange rate data.
    
    for days_back in range(max_days_back + 1):
        current_date = target_date - timedelta(days=days_back)
        try:
            # Try to fetch the exchange rate for the current date
            c.convert(1, 'GBP', 'USD', current_date)
            return current_date  # Return the closest valid date
        except:
            continue  # Try the previous day
    return None  # No valid date found within the range

# Function to convert GBP to USD using the closest available date
def convert_gbp_to_usd(price, date,fallback_rate = 1.25):
    try:
        # Try to fetch the exchange rate for the exact date
        return c.convert(price, 'GBP', 'USD', date)
    except:
        # If the exact date fails, find the closest valid date
        closest_date = get_closest_date(date)
        if closest_date:
            try:
            # print(f"No rate for {date}. Using closest available date: {closest_date}")
                return c.convert(price, 'GBP', 'USD', closest_date)
            except: 
                pass
        else:
            # print(f"No rate available for {date} or nearby dates.")
            return  price * fallback_rate  # Return None if no valid date is found
        
def convert_dataframe_to_usd(input_df):
    """
    Applies GBP to USD currency conversion row-wise to a DataFrame.
    Expects 'ds' (datetime) and 'price' (numeric) columns.
    Returns a DataFrame with 'unique_id', 'ds', and 'y' (price in USD) columns,
    or None if conversion fails significantly.
    """
    log.info(f"[Executor] Applying currency conversion to {len(input_df)} rows...") # Use log if setup, else print
    if input_df is None or input_df.empty:
        log.warning("[Executor] Input DataFrame is empty, cannot convert currency.")
        return None
    if 'price' not in input_df.columns or 'ds' not in input_df.columns:
         log.error("[Executor] Input DataFrame missing 'price' or 'ds' column for currency conversion.")
         return None

    # Ensure 'ds' is datetime before accessing .date()
    input_df['ds'] = pd.to_datetime(input_df['ds'])

    # Apply the row-wise conversion function
    # The convert_gbp_to_usd function handles finding the closest date if needed
    input_df['y'] = input_df.apply(
        lambda row: convert_gbp_to_usd(row['price'], row['ds'].date()), # Use .date() as convert_gbp_to_usd expects date object
        axis=1
    )

    # Check for conversion errors (NaNs)
    errors = input_df['y'].isnull().sum()
    if errors > 0:
        log.warning(f"[Executor] Currency conversion failed or resulted in NaN for {errors} rows.")
        # Decide how to handle errors: drop rows, fill with fallback, or return None
        # Dropping rows with failed conversion:
        input_df = input_df.dropna(subset=['y'])
        if input_df.empty:
             log.warning("[Executor] DataFrame empty after dropping rows with failed currency conversions.")
             return None

    # Remove original price column if conversion is successful
    if 'price' in input_df.columns:
        input_df = input_df.drop(columns=['price'])

    # Ensure 'unique_id' exists for Nixtla compatibility
    if 'unique_id' not in input_df.columns:
        # Assign a default unique_id if needed, or retrieve from context if available
        input_df['unique_id'] = 'item_forecast'

    log.info("[Executor] Currency conversion application finished.")
    # Return only the columns required for forecasting
    return input_df[['unique_id', 'ds', 'y']]


# ==================================================================
#                  === Step 6: Forecasting ===
# ==================================================================

async def generate_price_forecast(df, horizon):
    """
    Generates future price forecast using NixtlaClient.
    Handles synchronous Nixtla call within the async context.
    """
    if df is None or df.empty:
        log.warning("Cannot generate forecast: Input DataFrame is empty.")
        return None
    if not NIXTLA_API_KEY:
         log.error("Cannot generate forecast: Nixtla API Key not found in config or environment.")
         return None
    # instantiate the NixtlaClient class providing our authentication API key
    nixtla_client = NixtlaClient(api_key = NIXTLA_API_KEY)
    fcst_timegpt = nixtla_client.forecast(df = df,
                                      h=horizon,
                                      target_col = 'y',
                                      time_col='ds',
                                      level = [90, 95],
                                      finetune_loss= 'mse',
                                      finetune_steps= 30,
                                      finetune_depth= 3
                                      )
    fcst_timegpt = fcst_timegpt.rename(columns={'TimeGPT': 'y'})
    fcst_timegpt = fcst_timegpt[['ds', 'y']]  
    full_data = pd.concat([df, fcst_timegpt], ignore_index=True)
    return full_data[["ds" , "y"]]
# --- END OF FILE timegpt.py ---