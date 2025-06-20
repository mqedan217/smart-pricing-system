
# --- START OF FILE config.py ---

import os
import logging
from pymongo import MongoClient, errors as mongo_errors
from typing import Any, Optional, Dict, List, Set
from pathlib import Path

# --- Basic Logger for Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s[%(name)s]: %(message)s')
log = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
CONFIG_DB_NAME = "app_config" # databse name for config
CONFIG_COLLECTION_NAME = "settings" # Collection name for config
CONFIG_DOCUMENT_ID = "global_config" # Document ID for the config document


# --- Global variable to hold the loaded config ---
_config_cache: Optional[Dict[str, Any]] = None
_mongo_client_config: Optional[MongoClient] = None

# --- Function to Establish Initial DB Connection for Config ---
def _connect_to_config_db() -> Optional[MongoClient]:
    """Establishes the initial MongoDB connection to fetch config."""
    global _mongo_client_config
    if _mongo_client_config:
        try: _mongo_client_config.admin.command('ping'); return _mongo_client_config
        except mongo_errors.ConnectionFailure: log.warning("Existing config DB connection failed. Reconnecting."); _mongo_client_config = None

    mongo_uri_for_config = os.getenv("CONFIG_MONGO_URI", DEFAULT_MONGO_URI)
    log.info(f"Connecting to config MongoDB at: {mongo_uri_for_config}")
    try:
        client = MongoClient(mongo_uri_for_config, serverSelectionTimeoutMS=5000, appName="ConfigLoader")
        client.admin.command('ping')
        log.info("Successfully connected to MongoDB for configuration.")
        _mongo_client_config = client
        return client
    except mongo_errors.ConnectionFailure as e:
        log.critical(f"CRITICAL: Failed to connect to config MongoDB at {mongo_uri_for_config}. Error: {e}")
        return None
    except Exception as e:
        log.critical(f"CRITICAL: Unexpected error connecting to config MongoDB: {e}", exc_info=True)
        return None

# --- Function to Load Configuration ---
def load_config(force_reload: bool = False) -> bool:
    """Loads configuration from MongoDB into the _config_cache."""
    global _config_cache
    if _config_cache is not None and not force_reload:
        log.debug("Using cached configuration.")
        return True

    log.info(f"Loading configuration from MongoDB (DB: {CONFIG_DB_NAME}, Coll: {CONFIG_COLLECTION_NAME}, ID: {CONFIG_DOCUMENT_ID}).")
    client = _connect_to_config_db()
    if client is None: return False

    try:
        db = client[CONFIG_DB_NAME]
        collection = db[CONFIG_COLLECTION_NAME]
        config_doc = collection.find_one({"_id": CONFIG_DOCUMENT_ID})

        if config_doc:
            _config_cache = config_doc
            log.info("Configuration loaded successfully from MongoDB.")
            return True
        else:
            log.error(f"Config document _id='{CONFIG_DOCUMENT_ID}' not found in {CONFIG_DB_NAME}.{CONFIG_COLLECTION_NAME}.")
            _config_cache = {} # Fallback to empty dict
            return False
    except mongo_errors.PyMongoError as e:
        log.error(f"MongoDB error loading configuration: {e}", exc_info=True)
        _config_cache = {}
        return False
    except Exception as e:
        log.error(f"Unexpected error loading configuration: {e}", exc_info=True)
        _config_cache = {}
        return False

# --- Function to Safely Get Config Values ---
def get_config_value(keys: List[str], default: Any = None) -> Any:
    """Safely retrieves a nested value from the loaded MongoDB config cache."""
    global _config_cache
    if _config_cache is None:
        log.warning("Config cache is empty. Attempting initial load...")
        if not load_config():
             log.error(f"Initial config load failed. Using default for key '{'.'.join(keys)}': {default}")
             return default
        if _config_cache is None: # Should be {} on fallback
             log.error(f"Config cache still empty after load attempt. Using default for key '{'.'.join(keys)}': {default}")
             return default

    data = _config_cache
    try:
        for key in keys:
            if not isinstance(data, dict):
                 log.warning(f"Config path '{'.'.join(keys)}' traversal failed: encountered non-dictionary.")
                 return default
            data = data[key] # Raises KeyError if missing
        return data if data is not None else default
    except KeyError:
        if default is not None: log.warning(f"Config key '{'.'.join(keys)}' not found. Using default: {default}")
        # else: log.debug(f"Optional config key '{'.'.join(keys)}' not found.")
        return default
    except Exception as e:
        log.error(f"Error accessing config key '{'.'.join(keys)}': {e}")
        return default

# --- Load config on module import ---
if not load_config():
    log.critical("Initial configuration load from MongoDB FAILED. Application defaults will be used where available.")


# --- Define Accessor Variables (Reads from DB Config Document) ---

# Secrets
OPENAI_API_KEY: Optional[str] = get_config_value(["secrets", "openai_api_key"])
BRIGHTDATA_API_KEY: Optional[str] = get_config_value(["secrets", "brightdata_api_key"])
PROXY_USERNAME: Optional[str] = get_config_value(["secrets", "proxy_username"])
PROXY_PASSWORD: Optional[str] = get_config_value(["secrets", "proxy_password"])
FIREBASE_CRED_PATH : Optional[str] = get_config_value(["secrets", "firebase_cred_path"])
NIXTLA_API_KEY: Optional[str] = get_config_value(["secrets", "nixtla_api_key"]) 
# Database Names
APP_DB_NAME : str = get_config_value(["database", "app_db_name"], "appDB") # database name for app details (users, sessions, features)
FEATURES_COLLECTION_NAME : str = get_config_value(["database", "features_collection_name"], "features") 
USERS_COLLECTION_NAME : str = get_config_value(["database", "users_collection_name"], "users")
TOKENS_COLLECTION_NAME : str = get_config_value(["database", "tokens_collection_name"], "auth_tokens")
SESSIONS_COLLECTION_NAME : str = get_config_value(["database", "sessions_collection_name"], "sessions")
PRODUCTS_COLLECTION_NAME : str = get_config_value(["database", "products_collection_name"], "products")
CACHE_COLLECTION_NAME: str = get_config_value(["database", "cache_collection_name"], "questionnaire_cache")
PRICESPY_IDS_COLLECTION_NAME : str = get_config_value(["database","pricespy_collection_name"],"priceSpy_items_ids" )   
HISTORICAL_DATA_COLLECTION_NAME : str = get_config_value(["database", "historical_data_collection_name"], "pricespy_historical_data")  
SCHEDULED_TASKS_CONF_COLLECTION_NAME  : str = get_config_value(["database", "scheduled_tasks_config_collection_name"], "scheduled_tasks_config_collection")  
# Paths (resolve relative paths correctly)
try: _project_root = Path(__file__).resolve().parent
except NameError: _project_root = Path.cwd()

_yolo_rel_path = get_config_value(["paths", "yolo_model"], "yolov8x-oiv7.pt")
YOLO_MODEL_PATH: Path = Path(_yolo_rel_path) if Path(_yolo_rel_path).is_absolute() else _project_root / _yolo_rel_path

_output_rel_path = get_config_value(["paths", "output_images"], "output_images")
OUTPUT_IMAGES_PATH: Path = _project_root / _output_rel_path

_log_rel_path = get_config_value(["paths", "log_file"], "project_activity.log")
LOG_FILE_PATH: Path = _project_root / _log_rel_path

# Models
RESNET_WEIGHTS_STR: str = get_config_value(["models", "resnet_weights"], "DEFAULT")

# Image Recognition
_irrelevant_list = get_config_value(["image_recognition", "irrelevant_classes"], [])
IRRELEVANT_CLASSES: Set[str] = set(_irrelevant_list) if isinstance(_irrelevant_list, list) else set()
IMG_REC_NUM_MATCHES: int = get_config_value(["image_recognition", "match_params", "num_unique_results"], 3)
IMG_REC_K_MULTIPLIER: int = get_config_value(["image_recognition", "match_params", "search_k_multiplier"], 21)

# Questionnaire
MAX_QUESTIONS: int = get_config_value(["questionnaire", "max_questions"], 5)
OPENAI_QUESTION_MODEL: str = get_config_value(["questionnaire", "openai_question_model"], "gpt-4o-mini") # Add model config
OPENAI_QUESTION_TEMP: float = get_config_value(["questionnaire", "openai_question_temp"], 0.3)
OPENAI_QUESTION_TIMEOUT: float = get_config_value(["questionnaire", "openai_question_timeout"], 20.0)

# Scraper
SCRAPER_MAX_PRODUCTS: int = get_config_value(["scraper", "max_products_per_site"], 20)
SCRAPER_USE_PROXY: bool = get_config_value(["scraper", "use_proxy"], False)
SCRAPER_PROXY_HOST: str = get_config_value(["scraper", "proxy_host"], "res.proxy-sale.com")
SCRAPER_PROXY_PORTS: List[int] = get_config_value(["scraper", "proxy_ports"], list(range(10000, 10010)))
SCRAPER_NAVIGATION_RETRIES: int = get_config_value(["scraper", "navigation_retries"], 3)
SCRAPER_NAVIGATION_TIMEOUT_MS: float = get_config_value(["scraper", "navigation_timeout_ms"], 10.45000)
BRIGHTDATA_API_URL: str = get_config_value(["scraper", "brightdata_api_url"], "https://api.brightdata.com/request")
SCRAPER_GOOGLE_MAX_RESULTS: int = get_config_value(["scraper", "google_max_results"], 20)
SCRAPER_USER_AGENTS: List[str] = get_config_value(["scraper", "user_agents"], [])
UULE_LOCATIONS: Dict[str, str] = get_config_value(["scraper", "uule_locations"], {})

# --- Selectors (NEW Section in config) ---
SELECTORS: Dict[str, Dict[str, str]] = get_config_value(["selectors"], { # Default empty selectors
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
    }
        })
# Make sure structure like {'amazon': {'product': '...', 'title': '...'}, 'ebay': {...}} is in MongoDB

# Data Processing 
_price_conv = get_config_value(["data_processing", "price_conversion"], {})
AED_TO_USD: float = _price_conv.get("aed_to_usd", 0.27)
JOD_TO_USD: float = _price_conv.get("jod_to_usd", 1.41)
_w2v_defaults = {"vector_size": 1024, "window": 3, "min_count": 1, "workers": -1}
W2V_PARAMS: Dict[str, Any] = {**_w2v_defaults, **get_config_value(["data_processing", "word2vec"], {})}
RELEVANCE_MAX_RESULTS: int = get_config_value(["data_processing", "relevance_filtering", "max_results"], 8)
IQR_MULTIPLIER: float = get_config_value(["data_processing", "iqr_anomaly", "multiplier"], 1.0)

# Logging
LOGGING_LEVEL_STR: str = get_config_value(["logging", "level"], "INFO").upper()
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_STR, logging.INFO)

# Scheduler & TTL
PRICE_CHECK_INTERVAL_HOURS: int = get_config_value(["scheduler", "price_check_interval_hours"], 24)
SCRAPED_PRODUCT_TTL_SECONDS: int = get_config_value(["scheduler", "product_ttl_seconds"], 24 * 60 * 60) # Default 24h


# --- Optional: Function to re-load config if needed ---
def reload_configuration() -> bool:
    """Forces a reload of the configuration from MongoDB."""
    log.info("Attempting to force reload configuration from MongoDB...")
    return load_config(force_reload=True)

# --- Optional: Function to get the config MongoDB client if needed elsewhere ---
def get_config_db_client() -> Optional[MongoClient]:
     """Returns the client used for config loading, ensuring it's connected."""
     return _connect_to_config_db() # This function pings before returning

# --- END OF FILE config.py ---