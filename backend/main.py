# --- START OF FILE main.py ---

# =============================================================================
# ---  Imports & Initializing ---
# =============================================================================

# Standard Library Imports
import asyncio
import logging
import uuid
import sys
import os
import random
import time # For monotonic clock in rate limiting
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Dict, List
from collections import defaultdict # For rate limit locks
# Third-party Imports
import cv2
import numpy as np
import pandas as pd
from bson import ObjectId
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, Depends,Request, Header, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from passlib.context import CryptContext
import firebase_admin
from firebase_admin import credentials, messaging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.base import JobLookupError
# Project-specific Imports
import config # Your configuration file
import timegpt as price_predictor
from image_recognition import detect_products, find_best_match, crop_selected_object
from questionnaire import generate_next_question, identifyproduct
from scraper import scrape_all, store_input_features
from data_processing import finalresults, store_product_features
from user_model import UserCreate, AnswerRequest, UserLogin, SubscriptionRequest, FCMTokenRequest

# =============================================================================
# --- Logging Setup ---
# =============================================================================
try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILENAME = os.path.join(PROJECT_DIR, "project_activity.log")
except NameError:
     # __file__ might not be defined if run in certain interactive environments
     PROJECT_DIR = os.getcwd() # Fallback to current working directory
     LOG_FILENAME = os.path.join(PROJECT_DIR, "project_activity.log")
     print(f"Warning: Could not determine script directory, using CWD: {PROJECT_DIR}")
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
log_level = logging.INFO # Set desired level
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
# Clear existing handlers if any (important if script is re-run in same process)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    # Log confirmation - Use root_logger here before module-specific 'log' is defined
    root_logger.info(f"Logging initialized. Outputting to console and file: {LOG_FILENAME}")
except Exception as e:
    root_logger.error(f"CRITICAL: Failed to configure file logging to {LOG_FILENAME}: {e}", exc_info=True)
    root_logger.error("Logging will proceed to console only.")
# Get logger for this specific module AFTER root is configured
log = logging.getLogger(__name__)

# =============================================================================
# --- Global App Setup ---
# =============================================================================
# --- FastAPI App, Router, Auth Context ---
app = FastAPI()
router = APIRouter()
#  Creating context object to manage password hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Scheduler Globals ---
scheduler  = None
scheduler_db_client = None

# --- Firebase Global ---
firebase_initialized = False  # Firebase Initialization Flag 

# --- Rate Limiting Globals (In-Memory) ---
user_scrape_rate_limits: Dict[ObjectId, tuple[int, float]] = {}
# Stores asyncio.Lock per user_id to prevent race conditions
user_rate_limit_locks: Dict[ObjectId, asyncio.Lock] = defaultdict(asyncio.Lock)

# =============================================================================
# --- Lifespan Manager (Startup & Shutdown Events) ---
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    global scheduler, scheduler_db_client, firebase_initialized
    log.info("Application startup...") 
    db_client = None
    scheduler_db_client = None 
    scheduler = None 

    # 1. Initialize Firebase Admin SDK
    # ---------------------------------
    try:
        cred_path = config.FIREBASE_CRED_PATH
        if not firebase_admin._apps: # Initialize only if not already initialized
            if cred_path:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                log.info("✅ Firebase Admin SDK initialized successfully.")
                firebase_initialized = True
        else:
            log.info("Firebase Admin SDK already initialized.")
            firebase_initialized = True
    except FileNotFoundError: 
        log.error(f"❌ CRITICAL: Firebase key file not found at '{cred_path}'. Notifications will fail.")
        firebase_initialized = False
    except Exception as fb_err:
        log.error(f"❌ CRITICAL: Firebase Admin SDK initialization failed: {fb_err}", exc_info=True)
        firebase_initialized = False

    # 2. Connect to MongoDB & Attach Collections
    # ------------------------------------------
    try: 
        log.info("Connecting to MongoDB...")
        db_client = AsyncIOMotorClient(config.DEFAULT_MONGO_URI)
        # Verify connection
        await db_client.admin.command("ping")
        log.info("✅ MongoDB connection established")
        app.mongodb_client = db_client
        app.app_db = db_client[config.APP_DB_NAME]
    except Exception as e:
        log.critical(f"❌ CRITICAL: MongoDB connection failed: {e}", exc_info= True)
        raise RuntimeError(f"Failed to connect to MongoDB: {e}") from e
    # Initialize Database
    app.app_db = app.mongodb_client[config.APP_DB_NAME]
    # Initialize collections
    app.users_collection = app.app_db[config.USERS_COLLECTION_NAME] # to store users information
    app.auth_tokens_collection = app.app_db[config.TOKENS_COLLECTION_NAME] # to store users authentication tokens 
    app.sessions_collection = app.app_db[config.SESSIONS_COLLECTION_NAME] # to store session information 
    app.features_collection = app.app_db[config.FEATURES_COLLECTION_NAME] # to store the feature extracted from the image
    app.products_collection = app.app_db[config.PRODUCTS_COLLECTION_NAME] # to store all scraped posts 
    app.cache_collection = app.app_db[config.CACHE_COLLECTION_NAME] # to store questions 
    app.price_spy_ids_collection = app.app_db[config.PRICESPY_IDS_COLLECTION_NAME] # to store items IDs 
    app.pricespy_historical_data_collection = app.app_db[config.HISTORICAL_DATA_COLLECTION_NAME] # to store pricespy historical data
    app.scheduler_config_collection = app.app_db[config.SCHEDULED_TASKS_CONF_COLLECTION_NAME] 
    log.info("INFO: Database collections attached to app instance.")

    # 3. Create Indexes
    # ------------------
    try:  
        log.info("Checking/Creating MongoDB indexes...");
        # User/Auth
        # Index for finding users by email quickly
        await app.users_collection.create_index("email", unique=True)
        # Index for finding auth tokens quickly
        await app.auth_tokens_collection.create_index("auth_token", unique=True)
        # Optional: Index user_id in auth_tokens if you often delete tokens by user
        await app.auth_tokens_collection.create_index("user_id")
        log.info("INFO: Database indexes checked/created.")
        # Products Cache TTL
        ttl_seconds_products = getattr(config, "SCRAPED_PRODUCT_TTL_SECONDS", 24*60*60)
        await app.products_collection.create_index("createdAt", expireAfterSeconds=ttl_seconds_products);
        log.info(f"INFO: TTL index on 'products_collection.createdAt' checked/created ({ttl_seconds_products}s expiry).")
    except Exception as index_e:
        log.warning(f"WARN: Failed to create indexes: {index_e}")

    # 4. Startup Cleanup Task
    # -----------------------
    log.info("Running startup cleanup for old/unsubscribed sessions...")
    try:
        threshold = datetime.now(timezone.utc) - timedelta(hours=12)
        # Define the query for sessions to delete
        cleanup_query = {
            "notify_on_price_drop": False,
            "scrape_status": {"$in": ["completed", "error", "not_started"]},
            "$or": [
                    { "scrape_start_time": None },
                    { "scrape_start_time": { "$lt": threshold } }
                ]
            }
        delete_result = await app.sessions_collection.delete_many(cleanup_query)
        if delete_result.deleted_count > 0:
            log.info(f"Startup cleanup: Deleted {delete_result.deleted_count} unsubscribed session(s).")
        else:
            log.info("Startup cleanup: No unsubscribed sessions found to delete.")
    except Exception as cleanup_err:
        log.error(f"Error during startup session cleanup: {cleanup_err}", exc_info=True)

    # 5. Initialize and Start Scheduler
    # ---------------------------------
    log.info("Setting up scheduled tasks (In-Memory)...")
    scheduler = AsyncIOScheduler(timezone="UTC")
    try: 
        job_config_doc = await app.scheduler_config_collection.find_one({"_id": "price_drop_check_job_config"})
        if job_config_doc:
            log.info("Found existing job configuration in DB.")
            job_id = job_config_doc["_id"]
            interval_hours = job_config_doc.get("interval_hours", 24) # Get interval from DB
            db_config_job = job_config_doc.get("db_config_args", {}) # Get args from DB

            # Map function name string back to the actual function object
            job_func = globals().get(job_config_doc.get("function_name"))
            if job_func and callable(job_func):
                scheduler.add_job(
                    job_func, # The actual function object
                    trigger=IntervalTrigger(hours=interval_hours),
                    id=job_id,
                    name=job_config_doc.get("name", "Scheduled Job"),
                    replace_existing=True,
                    args=[db_config_job] # Pass arguments from DB
                     )
                log.info(f"Loaded and scheduled job '{job_id}' from DB config.")
            else:
                log.error(f"Could not find or call function '{job_config_doc.get('function_name')}' specified in job config {job_id}.")
        else:  # If no config in DB, add the default job and store its config
            log.info("No job configuration found in DB, setting up default...")
            job_id = "price_drop_check_job"
            db_config_job = {"uri": config.DEFAULT_MONGO_URI, "db_name": config.APP_DB_NAME}
            interval_hours = 24
            job_func = check_price_drops # Direct reference to the function

            scheduler.add_job(
                job_func,
                trigger=IntervalTrigger(hours=interval_hours),
                id=job_id, name="Check for price drops", replace_existing=True,
                args=[db_config_job]
                )
             # Store the config for next time
            await app.scheduler_config_collection.update_one(
                {"_id": job_id + "_config"},
                {"$set": {
                    "function_name": job_func.__name__, # Store function name as string
                    "interval_hours": interval_hours,
                    "db_config_args": db_config_job,
                    "name": "Check for price drops",
                    "last_configured": datetime.now(timezone.utc)
                     }},
                     upsert=True
                 )
            log.info(f"Added default job '{job_id}' and saved its configuration to DB.")
    except Exception as load_err:
        log.error(f"Error loading/adding scheduled jobs from DB: {load_err}", exc_info=True)
    scheduler.start()
    log.info(f"Scheduler started (In-Memory Store).")

    # --- Lifespan Yield ---
    yield  # Yield control back to FastAPI to start accept requests

    # --- Shutdown Logic ---
    # Close Scheduler DB when the API shuts down to avoid memory leak 
    log.info("Application shutdown...")
    if scheduler and scheduler.running:
        scheduler.shutdown()
        log.info("Scheduler shut down.")
    if scheduler_db_client:
         scheduler_db_client.close()
         log.info("Scheduler DB client closed.") # Close scheduler client
    app.mongodb_client.close()
    log.info("MongoDB connection closed.")

# Initialize FastAPI app with the lifespan context
app = FastAPI(lifespan=lifespan)

# =============================================================================
# === Helper Functions ===
# =============================================================================
# Create a new session in MongoDB.
async def create_session(session_data: dict, db: AsyncIOMotorCollection): 
    result = await db.insert_one(session_data)
    return result.inserted_id

# get session ID from MongoDB
async def get_session(session_id: str, db: AsyncIOMotorCollection):
    try:
        # Query the MongoDB sessions collection to find the session with the given session_id
        session = await db.find_one({"session_id": session_id})
        return session
    except Exception as e:
        log.error(f"Error fetching session: {e}")
        return None

async def get_user_by_email(email: str, db: AsyncIOMotorCollection):
    return await db.find_one({"email": email})

async def create_user(user_data: dict, db: AsyncIOMotorCollection):
    result = await db.insert_one(user_data)
    return result.inserted_id

# Creates an auth token when users login
async def create_auth_token(token_data: Dict[str, Any], db: AsyncIOMotorCollection): 
    result = await db.insert_one(token_data)
    return result.inserted_id

async def get_user_from_auth_token(request: Request, x_auth_token: Optional[str] = Header(None, alias="X-Auth-Token")):
    auth_tokens_collection = request.app.auth_tokens_collection
    users_collection = request.app.users_collection

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
    )

    if x_auth_token is None:
        log.debug("DEBUG: Missing X-Auth-Token header")
        raise credentials_exception

    # Look up auth token
    token_doc = await auth_tokens_collection.find_one({"auth_token": x_auth_token})

    # Validate token existence ONLY
    if not token_doc:
        log.debug(f"Auth Token not found: {x_auth_token}")
        raise credentials_exception

    # Find the associated user
    user_id = token_doc.get("user_id")
    if not user_id or not isinstance(user_id, ObjectId):
        log.error(f"Auth token {x_auth_token} missing or has invalid user_id!")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auth token data error")

    user = await users_collection.find_one({"_id": user_id})
    if not user:
        log.warning(f"User {user_id} for auth token {x_auth_token} not found! Deleting token.")
        await auth_tokens_collection.delete_one({"auth_token": x_auth_token})
        raise credentials_exception
    return user

# =============================================================================
# === Background Task Function Definitions ===
# =============================================================================

# --- Scraping Background Task ---
async def run_scrape_and_process(session_id, fullTitle, coreTitle_insideQuotes, category, features_vector, db_config):
    """
    Background task to perform scraping, processing, and storage for a session.
    Also stores successful results in the temporary products collection with TTL.

    """
    start_time = datetime.now(timezone.utc)
    log.info(f"--- BG TASK STARTED [{session_id}] @ {start_time.isoformat()} ---")

    # --- Initialize variables ---
    temp_client: Optional[AsyncIOMotorClient] = None
    sessions_collection: Optional[AsyncIOMotorCollection] = None
    products_collection: Optional[AsyncIOMotorCollection] = None
    features_collection: Optional[AsyncIOMotorCollection] = None
    final_status: str = "error" # Default to error unless explicitly completed
    error_details: Optional[str] = None
    processed_data: Optional[Dict] = None # Renamed for clarity

    try:
        # 1. Re-establish DB connection
        # =============================
        log.debug(f"BG TASK [{session_id}]: Step 1/6 - Connecting to Database...")
        if not db_config or not db_config.get("uri") or not db_config.get("db_name"):
             raise ValueError("Database configuration (uri, db_name) missing.")

        temp_client = AsyncIOMotorClient(
            db_config["uri"],
            serverSelectionTimeoutMS=15000,
            appName=f"BackgroundTask-{session_id[:8]}"
        )

        app_db = temp_client[db_config["db_name"]]
        sessions_collection = app_db[config.SESSIONS_COLLECTION_NAME]
        products_collection = app_db[config.PRODUCTS_COLLECTION_NAME]
        features_collection = app_db[config.FEATURES_COLLECTION_NAME]

        log.debug(f"BG TASK [{session_id}]: Step 1/6 - DB Connected.")

        # 2. Scrape Product Data
        # ======================
        scraped_data = None # Initialize scraped_data
        log.info(f"BG TASK [{session_id}]: Step 2/6 - Calling scrape_all with title: '{fullTitle}'...")
        # !! Ensure scrape_all is async def !!
        # Pass country code if needed, defaulting to 'jo' here
        scraped_data = await scrape_all(query=fullTitle)
        log.info(f"BG TASK [{session_id}]: Step 2/6 - scrape_all finished.")

        # Validate scraped_data (use the error handling from your snippet)
        if scraped_data is None or not isinstance(scraped_data, dict):
            log.warning(f"[{session_id}]: Web scraping function returned None. Proceeding with empty data.")
            scraped_data = {"amazon_products": [], "ebay_products": [], "google_shopping": [], "opensooq_products": []}
            error_details = "Scraping phase returned invalid/no data." # Log potential issue
        else:
            log.info(f"BG TASK [{session_id}]: Scraped Counts - Amazon: {len(scraped_data.get('amazon_products',[]))}, etc.") # Concise log

        # 3. Process Scraped Data
        # ========================
        processed_data = None # Initialize processed_data
        log.info(f"BG TASK [{session_id}]: Step 3/6 - Processing scraped data...")
        # !! Ensure finalresults is def (sync) !!
        processed_data = finalresults(scraped_data, fullTitle) 
        log.info(f"BG TASK [{session_id}]: Step 3/6 - Price processing finished.")

        # Validate processed_data 
        if processed_data is None or not isinstance(processed_data, dict):
            log.warning(f"[{session_id}]: finalresults returned None. Using default structure.")
            processed_data = { "amazon": {"products": [], "stats": {}}, "ebay": {"products": [], "stats": {}},
                               "google_shopping": {"products": [], "stats": {}}, "opensooq":{"products": [], "stats": []},
                               "overall_stats": {}, "related_products": [] }
            if error_details is None:
                error_details = "Processing phase returned invalid/no data."

        if error_details is None:
            final_status = "completed" # No errors recorded during scrape/process
        else:
            final_status = "error" # An error occurred somewhere
        log.info(f"BG SCRAPE [{session_id}]: Final Status Determined: {final_status}")

        # 4. Update both session_collection and products collection 
        # =======================================
        log.info(f"BG TASK [{session_id}]: Step 4/6 - Updating session collection and products collection...")
        # 4.1 updating products_collection 
        if processed_data and final_status == "completed" and products_collection is not None:
            try:
                product_doc = {
                    "item_title": fullTitle,
                    "scrape_results": processed_data, # Store the same results
                    "createdAt": datetime.now(timezone.utc) # Field for TTL index
                }
                insert_result = await products_collection.insert_one(product_doc)
                log.info(f"BG SCRAPE [{session_id}]: Successfully stored scrape results in products collection (ID: {insert_result.inserted_id}).")
            except Exception as prod_insert_err:
                # Log error but don't fail the whole task just for this
                log.error(f"BG SCRAPE [{session_id}]: Failed to store results in products collection: {prod_insert_err}", exc_info=True)
        # 4.2 updating user's session 
        update_payload = {
            "scrape_status": final_status,
            "scrape_results": processed_data, # Store the whole processed dict
            "scrape_error": None if error_details is None else error_details[:1000] # Store None if no error, else the message
        }
        await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": update_payload}
        )
        log.info(f"BG TASK [{session_id}]: Step 4/6 - Session results stored. Status: '{final_status}'.")

        # 5. Store Features (using the logic from your snippet)
        # ======================================================
        log.info(f"BG TASK [{session_id}]: Step 5/6 - Storing features...")
        if features_collection is None:
            log.warning(f"[{session_id}]: features_collection is None. Skipping feature storage.")
        else:
            # Use coreTitle_insideQuotes as the db key if it exists, otherwise fallback to fullTitle
            db_core_title = coreTitle_insideQuotes if coreTitle_insideQuotes else fullTitle
            if not db_core_title:
                log.warning(f"[{session_id}]: Cannot store features: No valid core title (core='{coreTitle_insideQuotes}', full='{fullTitle}').")
            else:
                 if isinstance(processed_data, dict):
                      # --- Store Product Features ---
                      log.debug(f"BG TASK [{session_id}]:   Calling store_product_features...")
                      # !! Ensure store_product_features is async def and accepts features_coll !!
                      await store_product_features(
                          category=category,
                          data=processed_data,
                          coreTitle=db_core_title,
                          features_coll=features_collection 
                      )
                      log.debug(f"BG TASK [{session_id}]:   store_product_features finished.")
                 else:
                      log.warning(f"WARNING [{session_id}]: Skipping scraped features storage: processed_data is not a dict.")

                 # --- Store Input Features ---
                 if features_vector is not None:
                     log.debug(f"BG TASK [{session_id}]:   Calling store_input_features...")
                     await store_input_features(
                         featuresVector=features_vector,
                         productkeyword=db_core_title,
                         category=category,
                         features_coll=features_collection, # Pass the async collection object
                     )
                     log.debug(f"BG TASK [{session_id}]:   store_input_features finished.")
                 else:
                     log.debug(f"BG TASK [{session_id}]:   Skipping store_input_features (no vector).")

            log.info(f"BG TASK [{session_id}]: Step 5/6 - Feature storage attempts finished.")

    # --- Exception Handling Block ---
    except Exception as e:
        final_status = "error" # Explicitly set error status
        error_details = f"Unhandled exception: {str(e)[:1000]}" # Limit error size
        log.error(f"!!! BG TASK ERROR [{session_id}]: {error_details} !!!" ,exc_info=True)

        # Attempt to update session status to error in DB
        if sessions_collection is not None: # Check if we have a collection object
             try:
                await sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "scrape_status": final_status,
                        "scrape_error": error_details,
                        # Store partial results if available from processing step
                        "scrape_results": processed_data if processed_data else {},
                    }}
                )
             except Exception as db_update_err:
                log.error(f"!!! BG TASK CRITICAL ERROR [{session_id}]: FAILED TO UPDATE SESSION STATUS TO ERROR: {db_update_err} !!!")
        else:
             log.error(f"!!! BG TASK CRITICAL ERROR [{session_id}]: Cannot update session status to error (DB connection failed?). Original error: {error_details} !!!")

    # --- always runs ---
    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        log.info(f"BG TASK [{session_id}]: Reached finally block. Final Status was: '{final_status}'. Duration: {duration}")

        # Update end time in session document
        if sessions_collection is not None: # Check if collection object exists
             try:
                await sessions_collection.update_one(
                     {"session_id": session_id},
                     {"$set": {"scrape_end_time": end_time}}
                 )
             except Exception as db_final_update_err:
                log.warning(f"!!! BG TASK WARNING [{session_id}]: Failed to update final scrape_end_time: {db_final_update_err} !!!")

        # Close DB connection
        if temp_client is not None:
            temp_client.close()
            log.debug(f"BG TASK [{session_id}]: DB Connection closed.")

    log.info(f"--- BG TASK FINISHED [{session_id}] @ {end_time.isoformat()} (Duration: {duration}) ---")

# --- Price Prediction Background Task ---
async def run_prediction_and_store(session_id: str, item_name: str, db_config: dict):
    """
    Background task for finding PriceSpy ID, fetching data, processing,
    forecasting using time_gpt module, and storing results in the session.
    """
    item_id: Optional[int] = None
    start_time = datetime.now(timezone.utc)
    log.info(f"--- BG PREDICT TASK STARTED [{session_id}] for item '{item_name}' ---")

    temp_client  = None
    sessions_collection  = None
    price_spy_ids_collection = None
    price_spy_historical_data_collection = None
    final_status: str = "prediction_error" # Default status
    error_details  = None
    final_results  = None # To store the JSON-serializable forecast

    # --- Helper function to update session status ---
    async def update_session_status(status: str, error: Optional[str] = None, results: Optional[Any] = None):
        nonlocal final_status, error_details, final_results # Allow modifying outer scope variables
        final_status = status
        error_details = str(error)[:1000] if error else None # Limit error length
        final_results = results # Store results if provided

        payload = {"prediction_status": final_status}
        if error_details:
            payload["prediction_error"] = error_details
        # Only store results if status is completed and results are available
        if results is not None and status == "prediction_completed":
            payload["prediction_results"] = results # Assume results are JSON serializable

        if sessions_collection is not None:
            try:
                await sessions_collection.update_one({"session_id": session_id}, {"$set": payload})
                log.info(f"Predict Task [{session_id}]: Session status updated to '{status}'.")
            except Exception as db_upd_err:
                log.error(f"Predict Task [{session_id}]: FAILED to update session status to '{status}': {db_upd_err}")
        else:
            log.error(f"Predict Task [{session_id}]: Cannot update session status (DB connection not available).")

    try:
        # 1. Re-establish DB connection
        log.debug(f"BG PREDICT [{session_id}]: Step 1/7 - Connecting to Database...")
        if not db_config or not db_config.get("uri") or not db_config.get("db_name"):
            raise ValueError("Database configuration (uri, db_name) missing.")

        temp_client = AsyncIOMotorClient(
            db_config["uri"],
            serverSelectionTimeoutMS=15000, # Adjust timeout if needed
            appName=f"BGPredictTask-{session_id[:8]}"
        )
        app_db = temp_client[db_config["db_name"]]
        sessions_collection = app_db[config.SESSIONS_COLLECTION_NAME]
        price_spy_ids_collection = app_db[config.PRICESPY_IDS_COLLECTION_NAME]
        price_spy_historical_data_collection = app_db[config.HISTORICAL_DATA_COLLECTION_NAME]
        log.info(f"BG PREDICT [{session_id}]: Step 1/7 - DB Connected.")

        # Get the current event loop for running sync code in executor
        loop = asyncio.get_running_loop()
        current_year_month = datetime.now(timezone.utc).strftime('%Y-%m')
        use_cached_data = False
        # 2. Find PriceSpy ID (Async - uses Playwright)
        await update_session_status("predicting_finding_id")
        log.info(f"BG PREDICT [{session_id}]: Step 2/7 - Checking cache/DB for PriceSpy ID for '{item_name}'...")
        id_doc = await price_spy_ids_collection.find_one({"_id": item_name})
        if id_doc and id_doc.get("item_id"):
            item_id = id_doc["item_id"]
            log.info(f"BG PREDICT [{session_id}]: Step 2/7 - Found cached PriceSpy ID in DB: {item_id}")
        else: 
            log.info(f"BG PREDICT [{session_id}]: Step 2/7 - ID not in DB cache. Finding via Playwright for '{item_name}'...")
            item_id = await price_predictor.find_pricespy_id(item_name) # Playwright call
            if item_id is None:
                # raise ValueError
                log.error(f"Could not find PriceSpy ID for '{item_name}' via Playwright.")
             
            else: # Store the newly found ID in the cache collection
                log.info(f"BG PREDICT [{session_id}]: Step 2/7 - Found PriceSpy ID: {item_id}") 
                try: 
                    await price_spy_ids_collection.update_one(
                        {"_id": item_name}, # Use core title as ID
                        {"$set": {
                            "item_id": item_id,
                            "last_updated": datetime.now(timezone.utc)
                         }},
                        upsert=True 
                    )
                    log.info(f"BG PREDICT [{session_id}]: Step 2/7 - Stored newly found ID {item_id} in DB cache for '{item_name}'.")
                except Exception as db_cache_err:
                    log.warning(f"BG PREDICT [{session_id}]: Failed to store newly found ID {item_id} in cache: {db_cache_err}")
        # 3. Fetch Historical Data (Async - uses httpx)
        log.info(f"BG PREDICT [{session_id}]: Step 3/7 - Checking cache/DB for PriceSpy historical data for '{item_name}'...")
        cached_doc = await price_spy_historical_data_collection.find_one({"_id": item_id})
        if cached_doc and cached_doc.get("last_data_point_month") == current_year_month and cached_doc.get("monthly_data"):
            log.info(f"BG PREDICT [{session_id}]: CACHE HIT - Valid historical data found for '{item_name}'.")
            try: 
                # Convert cached list of dicts back to DataFrame
                df_for_forecast = pd.DataFrame(cached_doc["monthly_data"])
                df_for_forecast['ds'] = pd.to_datetime(df_for_forecast['ds']) # Ensure ds is datetime
                use_cached_data = True
                df_usd = df_for_forecast
                log.info(f"BG PREDICT [{session_id}]: Successfully loaded cached data. Shape: {df_for_forecast.shape}")
            except Exception as load_cache_err:
                log.warning(f"BG PREDICT [{session_id}]: Error loading cached data for '{item_name}': {load_cache_err}. Will re-fetch.")
        else: 
            if cached_doc:
                log.info(f"BG PREDICT [{session_id}]: CACHE STALE for '{item_name}'. Cached month: {cached_doc.get('last_data_point_month')}, Current month: {current_year_month}. Will re-fetch.")
            else:
                log.info(f"BG PREDICT [{session_id}]: CACHE MISS for '{item_name}'. Will fetch fresh data.")
        if not use_cached_data:     
            await update_session_status("predicting_fetching_data")
            log.info(f"BG PREDICT [{session_id}]: Step 3/7 - Fetching historical data for ID {item_id}...")
            raw_data = await price_predictor.fetch_historical_data_async(item_id)
            if not raw_data:
                raise ValueError(f"Failed to fetch historical data for ID: {item_id}.")
            log.info(f"BG PREDICT [{session_id}]: Step 3/7 - Historical data fetched.")
            # 4. Process Data to DataFrame (Sync -> Executor)
            await update_session_status("predicting_processing")
            log.info(f"BG PREDICT [{session_id}]: Step 4/7 - Processing data into DataFrame...")
            df = await loop.run_in_executor(None, price_predictor.process_data_to_df, raw_data)
            if df is None or df.empty:
                raise ValueError("Processing fetched data failed: No DataFrame created.")
            log.info(f"BG PREDICT [{session_id}]: Step 4/7 - Data processed. DataFrame shape: {df.shape}")

            # 5. Fill Gaps (Sync -> Executor)
            await update_session_status("predicting_filling_gaps")
            log.info(f"BG PREDICT [{session_id}]: Step 5/7 - Filling time gaps in data...")
            df_filled = await loop.run_in_executor(None, price_predictor.gaps_filler, df.copy())
            if df_filled is None or df_filled.empty:
                raise ValueError("Gap filling failed or resulted in empty DataFrame.")
            # Ensure columns needed for next step
            df_filled = df_filled[['ds', 'price']]
            log.info(f"BG PREDICT [{session_id}]: Step 5/7 - Gaps filled. DataFrame shape: {df_filled.shape}")

            # 6. Convert Currency (Sync -> Executor)
            await update_session_status("predicting_converting")
            log.info(f"BG PREDICT [{session_id}]: Step 6/7 - Converting currency to USD...")
            # price_predictor.convert_dataframe_to_usd runs the sync apply internally
            df_usd = await loop.run_in_executor(None, price_predictor.convert_dataframe_to_usd, df_filled.copy())
            df_usd['year'] = df_usd['ds'].dt.year
            df_usd['month'] = df_usd['ds'].dt.month
            monthly_mode = df_usd.groupby(['year', 'month'])['y'].agg(lambda x: x.mode()[0] if not x.mode().empty else pd.NA)
            df_monthly_agg = pd.DataFrame(monthly_mode).reset_index()
            df_monthly_agg['ds'] = pd.to_datetime(df_monthly_agg[['year', 'month']].assign(day=1))
            df_monthly_agg = df_monthly_agg.drop(columns=['year', 'month']) #
            df_monthly_agg['y'] = df_monthly_agg['y'].astype(int)
            df_usd = df_monthly_agg
            df_usd = df_usd[['ds', 'y']]  
            if df_usd is None or df_usd.empty:
                raise ValueError("Currency conversion failed or resulted in empty DataFrame.")
            log.info(f"BG PREDICT [{session_id}]: Step 6/7 - Currency converted. DataFrame shape: {df_usd.shape}")
            try:
                last_date = df_usd['ds'].max()
                last_month_str = pd.to_datetime(last_date).strftime('%Y-%m')
                # Convert DataFrame to list of dicts for MongoDB storage
                monthly_data_list = df_usd.to_dict(orient='records')
                 # Convert datetime objects in list to native Python datetime for MongoDB
                for record in monthly_data_list:
                    if isinstance(record.get('ds'), pd.Timestamp):
                        record['ds'] = record['ds'].to_pydatetime().replace(tzinfo=timezone.utc) # Store as UTC

                cache_doc = {
                    "monthly_data": monthly_data_list,
                    "last_data_point_month": last_month_str, # Store YYYY-MM string
                    "last_updated": datetime.now(timezone.utc)
                }
                await price_spy_historical_data_collection.update_one(
                    {"_id": item_id}, # Use item_id as ID
                    {"$set": cache_doc},
                    upsert=True
                )
                log.info(f"BG PREDICT [{session_id}]: Stored/Updated aggregated data in historical cache for '{item_name}'.")
            except Exception as hist_cache_err:
                log.warning(f"BG PREDICT [{session_id}]: Failed to store aggregated data in cache: {hist_cache_err}")
        # 7. Generate Forecast (Async - Nixtla)
        await update_session_status("predicting_forecasting")
        log.info(f"BG PREDICT [{session_id}]: Step 7/7 - Generating price forecast (Horizon=12)...")
        # Assuming generate_price_forecast is correctly async or handles blocking internally
        forecast_df = await price_predictor.generate_price_forecast(df_usd, horizon=12) # e.g., 12 months
        
        serializable_result: Optional[list] = None # Initialize as list or None
        if forecast_df is None or forecast_df.empty:
            log.warning(f"Predict Task [{session_id}]: Forecast generation failed or returned empty. Storing historical data instead.")
            # Convert historical USD data to serializable format as fallback
            df_usd['ds'] = pd.to_datetime(df_usd['ds']).dt.strftime('%Y-%m-%d') # Format date for JSON
            serializable_result = df_usd.to_dict(orient='records')
            await update_session_status("prediction_completed", results=serializable_result, error="Forecast generation failed; historical data provided.")
        else:
            log.info(f"BG PREDICT [{session_id}]: Step 7/7 - Forecast generated. Shape: {forecast_df.shape}")
            # Convert forecast DataFrame to JSON serializable format (list of dicts)
            # Ensure 'ds' column is stringified
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.strftime('%Y-%m-%d')
            output_columns = ['ds', 'y'] 
            serializable_result = forecast_df[output_columns].to_dict(orient='records')
            # Update session with final completed status and results
            await update_session_status("prediction_completed", results=serializable_result)


    except Exception as e:
        error_msg = f"Unhandled exception in prediction task: {type(e).__name__} - {str(e)}"
        log.error(f"!!! BG PREDICT TASK ERROR [{session_id}] for item '{item_name}': {error_msg} !!!")
        # Attempt to update session status to error in DB
        await update_session_status("prediction_error", error=error_msg)

    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        log.info(f"--- BG PREDICT TASK FINISHING [{session_id}] Final Status: '{final_status}'. Duration: {duration} ---")

        # Attempt to update end time in session document, regardless of success/failure
        if sessions_collection is not None:
            try:
                await sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {"prediction_end_time": end_time}}
                )
            except Exception as db_final_update_err:
                log.warning(f"!!! BG PREDICT WARNING [{session_id}]: Failed to update final prediction_end_time: {db_final_update_err} !!!")

        # Close the temporary DB connection
        if temp_client is not None:
            temp_client.close()
            print(f"BG PREDICT [{session_id}]: DB Connection closed.")

    log.info(f"--- BG PREDICT TASK FINISHED [{session_id}] @ {end_time.isoformat()} (Duration: {duration}) ---")


# --- Notification Sending Function ---
async def send_price_drop_notification(user_id_obj: ObjectId, session_id: str, item_title: str, old_price: float, new_price: float, db_config):
    """
    Sends a push notification for a price drop using Firebase Admin SDK.
    
    """
    global firebase_initialized
    if not firebase_initialized:
        log.error(f"Firebase not initialized. Cannot send notification for session {session_id}.")
        return False
     
    log.info(f"Attempting to send price drop notification for session {session_id} to user {user_id_obj}")  
    fcm_token = None 
    temp_client = None 
    sent_successfully = False
    try:
        # 1. Establish temporary DB connection to get user's FCM token    
        temp_client = AsyncIOMotorClient(
            db_config["uri"],
            serverSelectionTimeoutMS = 5000,
            appName = "NotifUserLookup")
        user_db = temp_client[db_config["db_name"]]
        users_collection = user_db[config.USERS_COLLECTION_NAME]
        # Fetch FCM Token 
        user_doc = await users_collection.find_one(
            {"_id" : user_id_obj},
            {"fcm_token" : 1}
            )
        if user_doc and user_doc.get("fcm_token"):
            fcm_token = user_doc["fcm_token"]
            log.debug(f"Found FCM token for user {user_id_obj}")
        else: 
            log.warning(f"No FCM token found for user {user_id_obj}. Cannot send notification for session {session_id}.")
            return False # Cannot send without token
        # 3. Construct Notification Payload
        title = "Price Drop Alert! 📉"
        body = f"Price for '{item_title}' dropped from £{old_price:.2f} to £{new_price:.2f}!"
        # Data payload for handling click in Flutter
        data_payload = {
            "session_id": session_id,
            "click_action": "FLUTTER_NOTIFICATION_CLICK", # Standard for Flutter integration
            "screen": "session_results", # Example: tells Flutter which screen to open
            "new_price": str(new_price), # Send as string
            "item_title": item_title 
        }
        # 4. Create FCM Message object
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
                ),
            token=fcm_token,
            data=data_payload
        )
         # 5. Send the message (Run sync call in executor)
        loop = asyncio.get_running_loop()
        log.info(f"Sending FCM notification via executor for session {session_id}")
        log.info(f"Sending FCM notification to token {fcm_token[:10]}... for session {session_id}")
        # response = messaging.send(message)
        
        response = await loop.run_in_executor(None, messaging.send, message)
        log.info(f"Successfully sent FCM message for session {session_id}: {response}")
        log.info(f"Sent FCM message for {session_id}: {response}")
        sent_successfully = True
    except firebase_admin.exceptions.InvalidArgumentError as invalid_arg_err: 
        log.error(f"FCM Invalid Argument Error for {session_id}: {invalid_arg_err}", exc_info=False)
        error_str = str(invalid_arg_err).lower()
        if "registration token is not a valid fcm registration token" in error_str or \
        "mismatch" in error_str or "unregistered" in error_str:
            log.warning(f"Removing potentially invalid FCM token for user {user_id_obj}")
            if temp_client:  # Check if client exists before using collection
                try:
                    user_db = temp_client[db_config["db_name"]]
                    users_collection = user_db[config.USERS_COLLECTION_NAME]
                    await users_collection.update_one({"_id": user_id_obj}, {"$unset": {"fcm_token": ""}})
                except Exception as db_err:
                    log.error(f"Failed to remove invalid FCM token: {db_err}")
    except firebase_admin.FirebaseError as fcm_base_error:  # Catch other general Firebase errors
        log.error(f"FCM Base Error sending notification for session {session_id}: {fcm_base_error}", exc_info=True)
    except Exception as e:
        log.error(f"Unexpected error sending notification for session {session_id}: {e}", exc_info=True)
    finally:
        if temp_client:
            temp_client.close()
    return sent_successfully

async def check_price_drops(db_config: dict):

    """
    Scheduled job to check for price drops for subscribed items.
    Scrapes each unique subscribed item only ONCE per run.
    """
    log.info("--- Starting Scheduled Price Drop Check ---")
    start_time = datetime.now(timezone.utc)
    items_checked = 0
    sessions_updated = 0
    notifications_sent = 0
    temp_client: Optional[AsyncIOMotorClient] = None
    sessions_collection: Optional[AsyncIOMotorCollection] = None
    products_collection: Optional[AsyncIOMotorCollection] = None

    try:
        # 1. Connect to DB
        if not db_config or not db_config.get("uri") or not db_config.get("db_name"): raise ValueError("DB config missing.")
        temp_client = AsyncIOMotorClient(db_config["uri"], serverSelectionTimeoutMS=15000, appName="PriceDropCheckJobOpt")
        app_db = temp_client[db_config["db_name"]]
        sessions_collection = app_db[config.SESSIONS_COLLECTION_NAME]
        products_collection = app_db[config.PRODUCTS_COLLECTION_NAME]
        log.info("Optimized drop check: DB connected.")

        # 2. Get Unique List of Subscribed Item Titles (to avoid redundunt scrapping)
        pipeline = [
            {"$match": {"notify_on_price_drop": True, "scrape_status": "completed"}},
            {"$group": {"_id": "$fullTitle"}}, # Group by title to get unique ones
            {"$match": {"_id": {"$ne": None, "$ne": "Not Determined", "$ne": ""}}} # Filter out invalid titles
        ]
        unique_titles_cursor = sessions_collection.aggregate(pipeline)
        unique_item_titles = [doc["_id"] async for doc in unique_titles_cursor if doc.get("_id")]

        if not unique_item_titles:
            log.info("No subscribed items found needing a check.")
            return

        log.info(f"Found {len(unique_item_titles)} unique subscribed items to check.")

        # 3. Loop Through Unique Items
        for item_title in unique_item_titles:
            log.info(f"--- Checking item: '{item_title}...' ---")
            items_checked += 1
            new_processed_data = None
            new_min_price = None
            scrape_or_process_error = None
            now = datetime.now(timezone.utc) # Timestamp for this item check

            # Add inter-item delay
            inter_item_delay = random.uniform(5.0, 15.0)
            log.debug(f"Waiting {inter_item_delay:.1f}s before scraping '{item_title[:60]}...'")
            await asyncio.sleep(inter_item_delay)

            try:
                # 4. Scrape once per product 
                log.info(f"Scraping '{item_title[:60]}...'")
                # Assume default country or get from one session? Defaulting for now.
                new_scraped_data = await scrape_all(query=item_title)
                if new_scraped_data is None or not isinstance(new_scraped_data, dict):
                    raise ValueError("Re-scrape failed or returned invalid data.")

                # 5. Process once per product 
                log.info(f"Processing results for '{item_title}...'")
                new_processed_data = finalresults(new_scraped_data, item_title) 
                if new_processed_data is None or not isinstance(new_processed_data, dict):
                    raise ValueError("Re-processing failed or returned invalid data.")

                new_min_price = new_processed_data.get("overall_stats", {}).get("min_price")
                if new_min_price is None:
                    raise ValueError("Could not find minimum price in processed data.")

                log.info(f"Successfully scraped/processed '{item_title}...'. New Min Price: {new_min_price}")

                # 6. Update Product Cache
                if products_collection is not None:
                    try:
                        # Get core title if possible - maybe fetch one session for it?
                        # For simplicity, we might omit core title here or query it separately
                        product_doc = {
                            "item_title": item_title,
                            "scrape_results": new_processed_data,
                            "createdAt": now
                        }
                        await products_collection.update_one(
                            {"item_title": item_title}, {"$set": product_doc}, upsert=True
                        )
                        log.info(f"Updated products cache for '{item_title[:60]}...'.")
                    except Exception as prod_store_err:
                        log.error(f"Failed to update products cache for '{item_title}': {prod_store_err}", exc_info=True) # Log but continue

            except Exception as item_check_err:
                scrape_or_process_error = str(item_check_err)[:500]
                log.error(f"Error scraping/processing item '{item_title}': {scrape_or_process_error}", exc_info=True)
                # Continue to the next item_title in the main loop

            # 7. Find and Update Affected Sessions for THIS item_title
            if scrape_or_process_error is None and new_min_price is not None:
                # Find all sessions subscribed to this specific item
                session_query = {
                    "fullTitle": item_title,
                    "notify_on_price_drop": True,
                    "scrape_status": "completed"
                }
                affected_sessions = sessions_collection.find(session_query)

                async for session in affected_sessions:
                    session_id = session.get("session_id")
                    user_id = session.get("user_id")
                    old_results = session.get("scrape_results")
                    core_title = session.get("coreTitle_insideQuotes")
                    if not (session_id and user_id and old_results): continue # Skip incomplete sessions

                    old_min_price = old_results.get("overall_stats", {}).get("min_price")
                    if old_min_price is None: log.warning(f"Session {session_id} missing old min price."); continue

                    session_update = {"last_price_check_time": now} # Always update check time
                    price_dropped = False
                    try:
                        # Compare numeric prices
                        if float(new_min_price) < float(old_min_price):
                             # Apply threshold check if desired
                             drop_rate = float(new_min_price) / float(old_min_price)
                             if drop_rate <= 0.95: # Example: 5% drop needed
                                  price_dropped = True
                    except (TypeError, ValueError, ZeroDivisionError):
                        log.warning(f"Price comparison failed for session {session_id}"); continue

                    if price_dropped:
                        log.info(f"PRICE DROP for session {session_id} ({item_title[:50]}...)! Old: {old_min_price}, New: {new_min_price}")
                        session_update["scrape_results"] = new_processed_data # Update results
                        session_update["price_drop_detected_time"] = now
                        session_update["last_notified_price"] = new_min_price
                        notifications_sent += 1
                        try: # Send notification
                           await send_price_drop_notification(user_id_obj=user_id, session_id=session_id, item_title=core_title or item_title, old_price=float(old_min_price), new_price=float(new_min_price), db_config=db_config)
                        except Exception as notify_err: log.error(f"Failed to send notification for {session_id}: {notify_err}", exc_info=True) # Log notify error but still update session
                    else: # No drop for this specific session
                        pass # Only update last_check_time

                    try: # Update individual session
                       await sessions_collection.update_one({"session_id": session_id}, {"$set": session_update})
                       sessions_updated += 1
                    except Exception as session_update_err:
                        log.error(f"Failed to update session {session_id}: {session_update_err}", exc_info=True)

            else: # If scrape/process failed for this item_title
                 log.warning(f"Skipping session updates for item '{item_title}' due to scrape/process error: {scrape_or_process_error}")
                 # Optionally update all sessions for this item with the check time and error?
                 await sessions_collection.update_many(
                     {"fullTitle": item_title, "notify_on_price_drop": True},
                     {"$set": {"last_price_check_time": now, "last_check_error": f"Item scrape/process failed: {scrape_or_process_error}"}}
                 )

        # End of loop through unique items
        log.info(f"--- Finished Price Drop Check ({items_checked} items checked, {sessions_updated} sessions updated, {notifications_sent} notifications sent). Duration: {datetime.now(timezone.utc) - start_time} ---")

    except Exception as job_err:
        log.error(f"!!! CRITICAL ERROR in Optimized Price Drop Job: {job_err} !!!", exc_info=True)
    finally:
        if temp_client:
            temp_client.close()
            log.info("Optimized drop check: DB connection closed.")


# =============================================================================
# === API Routes Definition ===
# =============================================================================

# -------------------------------------------
# --- Auth & User Routes ---
# -------------------------------------------
@router.post("/signup/", status_code=status.HTTP_201_CREATED)
async def signup(user: UserCreate, request : Request):
    users_collection = request.app.users_collection
    existing_user = await get_user_by_email(user.email, users_collection)
    if existing_user: # check if the email already exists 
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed_password = pwd_context.hash(user.password) # Hashing the password before stroing it in DB 
    user_data = {
        "email": user.email,
        "password": hashed_password
    }
    try:
        await create_user(user_data, users_collection)  # Store user information in DB
    except Exception as e:
         log.error(f"ERROR: Failed to insert user {user.email}: {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not create user account.")

    return {"message": "User created successfully"} # this is what is returned to the app


@router.post("/login/" , response_model=UserLogin)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    users_collection = request.app.users_collection
    auth_tokens_collection = request.app.auth_tokens_collection
    log.info(f"Received login attempt for: {form_data.username}")

    user = await get_user_by_email(form_data.username, users_collection)  # check if the email already registered in DB

    if not user or not pwd_context.verify(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # raise 401 for bad credintials 
            detail="Incorrect username or password",
        )
    print(f"Successfully authenticated user: {user['email']}")  # Verify user lookup
    auth_token = str(uuid.uuid4()) # creating auth token randomly 

    token_data = {
        "auth_token": auth_token,
        "user_id": user["_id"],
        "email": user["email"] 
    }

    try:
        await auth_tokens_collection.delete_many({"user_id": user["_id"]})
        # Insert the new auth token
        await create_auth_token(token_data, auth_tokens_collection)
        log.debug(f"Created auth token {auth_token} for user {user['email']}")
    except Exception as e:
        log.error(f"ERROR: Failed saving auth token for {user['email']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error."
        )

     # Return Auth Token to Flutter
    return {
        "message": "Login successful",
        "auth_token": auth_token, # The generated token
        "email": user["email"]
    }


@router.post("/logout/")
async def logout( request: Request, x_auth_token: str | None = Header(None, alias="X-Auth-Token")):
    if x_auth_token is None:
        # Even if header is missing, client wants to log out locally.
        log.info("Logout attempt with missing X-Auth-Token header.")
        return {"message": "Logout successful (no token provided)"}

    auth_tokens_collection = request.app.auth_tokens_collection
    # --- This delete is now the ONLY way the token becomes invalid ---
    result = await auth_tokens_collection.delete_one({"auth_token": x_auth_token})

    if result.deleted_count == 1:
        log.info(f"Logged out and deleted auth token: {x_auth_token}")
        return {"message": "Logout successful"}
    else:
        # Token provided was already invalid or deleted
        log.info(f"Logout attempt for non-existent auth token: {x_auth_token}")
        return {"message": "Logout successful (token not found)"}

@router.put("/users/me/fcm-token")
async def register_fcm_token(
    token_request: FCMTokenRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    """Stores or updates the user's FCM registration token."""
    users_collection = request.app.users_collection
    user_id = current_user["_id"]
    new_token = token_request.fcm_token
    if not new_token:
        raise HTTPException(400, "fcm_token cannot be empty.")
    log.info(f"Request to update FCM token for user {user_id}")
    try: # Update user document 
        result = await users_collection.update_one(
            {"_id": user_id},
            {"$set": {"fcm_token": new_token, "fcm_token_updated_at": datetime.now(timezone.utc)}}
        )
        if result.matched_count == 0:
             # Should not happen if get_user_from_auth_token works
             log.error(f"Failed to find user {user_id} to update FCM token.")
             raise HTTPException(status_code=404, detail="User not found during token update.")

        log.info(f"Successfully updated FCM token for user {user_id}")
        return {"message": "FCM token registered successfully."}
    except Exception as e:
        log.error(f"Failed to store FCM token for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store FCM token.")
    
# -------------------------------------------
# --- Core Workflow Routes ---
# -------------------------------------------
@router.post("/detect/")
async def detect_objects(
    request: Request,
    file: UploadFile = File(...),  # file is the variable that will contain the uploaded image file.
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)): 
    user_id = current_user["_id"] # user_id will be available if the token was valid
    user_email = current_user["email"]
    sessions_collection = request.app.sessions_collection
    features_collection = request.app.features_collection
    cache_collection = request.app.cache_collection
    try:
        # --- Step 1: Image Input and Reading ---
        # Read the uploaded image file as an OpenCV image
        image_bytes = await file.read() # This reads the contents of the uploaded file, which will contain the raw bytes of the image.
        image_np = np.frombuffer(image_bytes, np.uint8) # This converts the raw bytes into a NumPy array, that OpenCV can work with.
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR) #  This decodes the byte array (image_np) into an image object that can be processed further. cv2.IMREAD_COLOR tells OpenCV to read the image as a color image.
        
        if image is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image format")
        
        # --- Step 2: Object Detection and Cropping ---
        # Detect products in the image
        top_detections = detect_products(image)
        if top_detections is None or not top_detections.get("class"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No relevant products detected or class missing")
        
        category = top_detections.get("class")
        cropped_image =crop_selected_object(image,top_detections)
        if cropped_image is None: 
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to crop detected object")
        # --- Step 3: Feature Matching ---
        match_data = await find_best_match(
            features_collection,
            cropped_image=cropped_image,
            category=category, num_unique_results=3 )
        
        features_vector = None
        top_matches = None
        original_match_titles = [] # Store original titles here
        if match_data:
            features_vector = match_data.get("features_vector")
            top_matches = match_data.get("results") 
             # check if there is a similar item in DB so there is no need for questionnaire
            if top_matches:
                original_match_titles = [f"{match.get('title', 'N/A')[:75]}{'...' if len(match.get('title', 'N/A')) > 75 else ''}" for match in top_matches]
        
        # Preparing the session 
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "user_email" : user_email,
            "status" : "starting", # overall session status
            "initial_matches_found": bool(top_matches),
            "ready_title" : False,
            "category": category,
            "potential_titles": original_match_titles,
            "current_question": 1,
            "features_vector" : features_vector,
            "fullTitle": "Not Determined",
            "coreTitle_insideQuotes": "Not Determined",
            "notify_on_price_drop": False, 
            "last_price_check_time": None,
            "last_notified_price": None, 
            "price_drop_detected_time": None ,
            "prediction_status" : "not_started",
            "prediction_results" : None,
            "prediction_start_time": None,
            "prediction_end_time": None,
            "scrape_status": "not_started",
             "scrape_source" : "not_determined",
            "scrape_results": None,
            "scrape_error": None,
            "scrape_start_time": None,
            "scrape_end_time": None
                }
        # Store session in MongoDB
        session = await create_session(session_data,db=sessions_collection)
        
        if top_matches: 
            display_options =  [f"{title[:75]}{'...' if len(title) > 75 else ''}" for title in original_match_titles]
            display_options.append("None of the above")
            await sessions_collection.update_one(
                {"session_id" : session_id},
                {"$set": {"status": "potential_matches"}}
            )
            return JSONResponse(content={
                "status" : "potential_matches",
                "session_id" : session_id,
                "potentials" : display_options
            })
        else: 
            #if there is no similiar product found then the questionnarie is staritng        
            first_question = await generate_next_question(cache_collection,category, [])
            await sessions_collection.update_one(
                {"session_id": session_id},
                 {"$set": {
                    "status": "questionnaire_started",
                    "current_question_details": first_question # Store question details
                 }}
            )
            # append none of the above option for all questions so the user can specify custom answer
            choices_list = first_question.get('choices', [])
            choices_list.append("None of the above")
        # Return the first question with its choices
            return JSONResponse(content={
                "status": "questionnaire_started",
                "session_id": session_id,
                "question": first_question['question'],
                "choices": choices_list ,
                "question_number": 1,
                "total_questions": config.MAX_QUESTIONS
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer/")
async def handle_answer(
    request: Request,
    answer_req: AnswerRequest, # Use Pydantic model for session_id and answer
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    session_id = answer_req.session_id
    user_answer = answer_req.answer
    user_id = current_user["_id"]
    # Get collections
    sessions_collection = request.app.sessions_collection
    cache_collection = request.app.cache_collection

    # --- Step 1: Fetch Session ---
    session = await get_session(session_id, sessions_collection)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # --- Step 2: Validate Session & User ---
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")

    session_status = session.get("status")
    category = session.get("category", "Unknown") # Get category for questionnaire
    current_answers = session.get("answers", []) # Get current answers list

    try:
        # --- Step 3: Process Answer based on Status ---
        if session_status == "potential_matches":
            potential_titles = session.get("potential_titles", []) 
            if user_answer != "None of the above" and user_answer in potential_titles:
                # --- User Confirmed a Match ---
                await sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "status": "specs", # Skip questionnaire and go to specs question
                        "coreTitle_insideQuotes" : user_answer,
                        "ready_title" : True
                    }}
                )

                return JSONResponse(content={
                    "status": "specs", # before scraping just taking some specs from the user 
                    "session_id": session_id,
                    "specs_message": "Add any other specifications (e.g., color, condition) or press Enter",
                })  
            else:
                # --- User selected "None of the above" or invalid choice ---
                current_answers = [] # Reset answers for questionnaire start

                # Generate the first question
                first_question = await generate_next_question(cache_collection, category, []) # Pass empty answers
                first_question.get("answers")
                if not first_question or "[Error]" in first_question.get("question", ""):
                    await sessions_collection.update_one({"session_id": session_id}, {"$set": {"status": "error_questionnaire_start"}})
                    raise HTTPException(status_code=500, detail="Could not generate the first question.")

                # Update session status and details
                await sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "answers": [], # Ensure answers are reset
                        "current_question": 1,
                        "current_question_details": first_question,
                        "status": "questionnaire_started" # Explicitly start questionnaire
                    }}
                )
                # append none of the above option for all questions so the user can specify custom answer
                choices_list = first_question.get('choices', [])
                choices_list.append("None of the above")
                # Return the first question
                return JSONResponse(content={
                    "status": "questionnaire_started",
                    "session_id": session_id,
                    "question": first_question['question'],
                    "choices": choices_list,
                    "question_number": 1,
                    "total_questions": config.MAX_QUESTIONS
                })

        elif session_status in "questionnaire_started":
            # Scenario B: User is answering a questionnaire question
            # Append the new answer
            current_answers.append(user_answer)
            next_question_num = len(current_answers) + 1

            # Check if max questions reached
            if len(current_answers) >= config.MAX_QUESTIONS:
                # --- Max Questions Reached: Identify Product ---
                title_result = await identifyproduct(cache_collection, category, current_answers, specs="", readyFromDatabase=False)
                final_title = title_result.get("product_title", "Could not determine title")

                await sessions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "answers": current_answers, # Store final answers
                        "coreTitle_insideQuotes": final_title,
                        "status": "specs",
                        "current_question": next_question_num # Mark as past last question
                    }}
                )

                return JSONResponse(content={
                    "status": "specs", # before scraping just taking some specs from the user 
                    "session_id": session_id,
                    "specs_message": "Add any other specifications (e.g., color, condition) or press Enter",
                })  
            else:
                # --- More Questions Needed: Generate Next ---
                # Call generate_next_question (async) with updated answers
                next_q_data = await generate_next_question(cache_collection, category, current_answers)

                if not next_q_data or "[Error]" in next_q_data.get("question", ""):
                    await sessions_collection.update_one({"session_id": session_id}, {"$set": {"status": "error_questionnaire_cont"}})
                    raise HTTPException(status_code=500, detail="Could not generate the next question.")

                # Update session: store new answer list, next question details
                await sessions_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "answers": current_answers, # Save the appended answer
                            "current_question_details": next_q_data,
                            "status": "questionnaire_started" # Still in progress
                        },
                        "$inc": {"current_question": 1} # Increment counter atomically
                    }
                )
                # append none of the above option for all questions so the user can specify custom answer
                choices_list = next_q_data.get('choices', [])
                choices_list.append("None of the above")
                # Return next question
                return JSONResponse(content={
                    "status": "questionnaire_started",
                    "session_id": session_id,
                    "question": next_q_data['question'],
                    "choices": choices_list,
                    "question_number": next_question_num,
                    "total_questions": config.MAX_QUESTIONS
                })
        # End of questionnaire or a potential match is chosen 
        elif session_status in "specs":
            specs = user_answer
            ready_title = session.get("ready_title")
            title_input = session.get("coreTitle_insideQuotes") if ready_title else current_answers
            title_result = await identifyproduct(cache_collection,category, title_input, specs, ready_title)
            if not title_result or not title_result.get('product_title'):
                print("Failed to generate product title. Constructing fallback.")
                core_fallback = str(title_input) if isinstance(title_input, str) else " ".join(title_input)
                full_fallback = f'"{core_fallback}" {specs}'.strip()
                coreTitle_insideQuotes = core_fallback; fullTitle = full_fallback
            else:
                coreTitle_insideQuotes = title_result.get('core_title', '')
                fullTitle = title_result.get('product_title', '')
                if not coreTitle_insideQuotes and fullTitle and '"' in fullTitle and fullTitle.count('"') >= 2:
                    try: 
                        coreTitle_insideQuotes = fullTitle.split('"')[1]
                    except IndexError:
                        coreTitle_insideQuotes = fullTitle

            if not fullTitle: 
                print(f"CRITICAL: Error during title identification: . Cannot proceed.") 
                return
            await sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"status": "done",
                "coreTitle_insideQuotes" : coreTitle_insideQuotes,
                "fullTitle": fullTitle
                }})
            return JSONResponse(content={
                    "status": "done",
                    "session_id": session_id,
                    "your product title is" : coreTitle_insideQuotes
                })
        else:
            # Scenario C: Session is in an unexpected state
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Session is not currently expecting an answer (status: {session_status}).")

    except HTTPException as he:
        raise he
    except Exception as e:
        await sessions_collection.update_one({"session_id": session_id}, {"$set": {"status": "error_processing_answer"}})
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the answer.")
    

# -------------------------------------------
# --- Scraping Routes ---
# -------------------------------------------
@router.post("/scrape/{session_id}", status_code=status.HTTP_202_ACCEPTED)
async def trigger_scraping(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks, # Inject FastAPI's BackgroundTasks
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    """
    Checks for cached results first. If not found or expired,
    triggers the background scraping process for a completed session.
    Returns existing results immediately if cache hit.

    """

    sessions_collection = request.app.sessions_collection
    products_collection = request.app.products_collection
    # --- Rate Limit Check ---
    user_id = current_user["_id"] # Get the user ID for rate limiting
    RATE_LIMIT = 2  # Max requests
    RATE_WINDOW = 60 # Seconds
    user_lock = user_rate_limit_locks[user_id]
    async with user_lock: # Ensure atomic check/update for this user
        current_time = time.monotonic() # Use monotonic clock for intervals
        request_count, window_start = user_scrape_rate_limits.get(user_id, (0, 0.0))
        if current_time - window_start > RATE_WINDOW:
            # Window expired, reset count and start new window
            log.debug(f"Rate limit window reset for user {user_id}")
            user_scrape_rate_limits[user_id] = (1, current_time) # Start count at 1
        else:
            # Within window, check count
            if request_count >= RATE_LIMIT:
                log.warning(f"Rate limit exceeded for user {user_id}. Count: {request_count}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {int(RATE_WINDOW - (current_time - window_start))} seconds."
                )
            else:
                # Increment count within window
                user_scrape_rate_limits[user_id] = (request_count + 1, window_start)
                log.debug(f"Rate limit check passed for user {user_id}. Count: {request_count + 1}")\
    # --- End Rate Limit Check ---


    # --- Fetch and Validate Session ---
    session = await get_session(session_id, sessions_collection)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")
    fullTitle = session.get("fullTitle")
    if not fullTitle:
        raise HTTPException(400, "fullTitle not found in session.")
    
    log.info(f"Checking products cache for title: '{fullTitle}'")
    cache_hit = None
    try:
        ttl_seconds = 24 * 60 * 60  # Calculate expiry threshold (e.g., data less than 24 hours old)
        expiry_threshold = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        # Find the *most recent* valid entry for this title
        cache_hit = await products_collection.find_one(
            {
                "item_title": fullTitle, # Find by the exact title scraped before
                "createdAt": {"$gte": expiry_threshold} # Ensure it's not expired
            },
            sort=[("createdAt", -1)] # Get the latest one if multiple match
        )
        if cache_hit and cache_hit.get("scrape_results"):
            log.info(f"CACHE HIT for session {session_id}, title '{fullTitle}'. Returning cached results.")
            cached_results = cache_hit.get("scrape_results")
            cached_time = cache_hit.get("createdAt")
            # --- Immediately update session with cached results and return ---
            await sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "scrape_status": "completed", # Mark as completed
                    "scrape_results": cached_results,
                    "scrape_error": None,
                    "scrape_start_time": cached_time, # Use cache time as start/end
                    "scrape_end_time": cached_time,
                    "scrape_source": "cache" # Indicate source
                }}
            )
            # Return status and results directly
            return JSONResponse(
                status_code=status.HTTP_200_OK, # Return 200 since results are ready
                content={
                    "session_id": session_id,
                    "scrape_status": "completed",
                    "results": cached_results,
                }
            )
        else:
            log.info(f"CACHE MISS for session {session_id}, title '{fullTitle}'. Proceeding to scrape.")
            
    except Exception as cache_check_err:
        log.error(f"Error checking products cache for session {session_id}: {cache_check_err}", exc_info=True)
    
    # --- Check if session is ready for scraping ---
    if session.get("status") != "done":
         raise HTTPException(
             status_code=status.HTTP_409_CONFLICT,
             detail=f"Session is not ready for scraping. Current status: {session.get('status')}"
         )

    # --- Check if scraping already running/done ---
    current_scrape_status = session.get("scrape_status", "not_started")
    if current_scrape_status in ["processing", "completed", "completed_placeholder"]: # Add placeholder if using temp func
         print(f"Scraping trigger attempt ignored for session {session_id}, status: {current_scrape_status}")
         # You could return the current status here instead of raising an error
         # return await get_scraping_status(session_id, request, current_user)
         raise HTTPException(
             status_code=status.HTTP_409_CONFLICT,
             detail=f"Scraping already '{current_scrape_status}' for session {session_id}."
         )

    # --- Get necessary data from session ---
    fullTitle = session.get("fullTitle")
    coreTitle_insideQuotes = session.get("coreTitle_insideQuotes")
    category = session.get("category")
    features_vector = session.get("features_vector") # Make sure this was stored in /detect

    if not fullTitle:
        await sessions_collection.update_one( {"session_id": session_id}, {"$set": {"scrape_status": "error", "scrape_error": "Missing fullTitle"}} )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="fullTitle not found in session.")

    # --- Update Session Status (Mark as Processing) ---
    log.info(f"Triggering scraping for session {session_id}, title: '{fullTitle}'")
    await sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {"scrape_status": "processing", "scrape_start_time": datetime.now(timezone.utc), "scrape_error": None, "scrape_source": "scrapping"}} # Clear previous errors
    )

    # --- Add Scraping Task to Background ---
    db_config = {"uri": config.DEFAULT_MONGO_URI, "db_name": config.APP_DB_NAME}
    background_tasks.add_task(
        run_scrape_and_process, # The async function doing the work
        session_id=session_id,
        fullTitle=fullTitle,
        coreTitle_insideQuotes=coreTitle_insideQuotes,
        category=category,
        features_vector=features_vector,
        db_config=db_config
    )

    log.info(f"Background task added for session {session_id}")
    return {"message": "Scraping process initiated."}

@router.get("/scrape-status/{session_id}")
async def get_scraping_status(
    session_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    """
    Checks the status of the scraping process and returns results if completed.
    """
    sessions_collection = request.app.sessions_collection

    # --- Fetch and Validate Session ---
    session = await get_session(session_id, sessions_collection)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")

    scrape_status = session.get("scrape_status", "not_started")
    response_data = {"session_id": session_id, "scrape_status": scrape_status}

    # Add results or error message based on status
    if scrape_status == "completed" or scrape_status == "completed_placeholder": # Include placeholder if using temp func
        # IMPORTANT: Decide what structure 'scrape_results' will have when stored by the background task
        response_data["results"] = session.get("scrape_results", {}) # Assume results are stored under this key
    elif scrape_status == "error":
        response_data["error_message"] = session.get("scrape_error", "Unknown error")

    log.debug(f"Checked scrape status for session {session_id}: {scrape_status}")
    return response_data

# -------------------------------------------
# --- Price Prediction Routes ---
# -------------------------------------------
@router.post("/predict-prices/{session_id}", status_code=status.HTTP_202_ACCEPTED)
async def trigger_price_prediction(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks, # Inject FastAPI's BackgroundTasks
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    """
    Triggers the background price prediction process for a completed session.
    """
    sessions_collection = request.app.sessions_collection

    # --- Fetch and Validate Session ---
    session = await get_session(session_id, sessions_collection)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")

    # --- Check if session is ready (questionnaire/detection done) ---
    if session.get("status") != "done":
         raise HTTPException(
             status_code=status.HTTP_409_CONFLICT,
             detail=f"Session is not ready for prediction. Current status: {session.get('status')}"
         )

    # --- Check if prediction already running/done ---
    current_predict_status = session.get("prediction_status", "not_started")
    # Add more detailed processing statuses if used in background task
    busy_statuses = [
        "processing", "predicting_finding_id", "predicting_fetching_data",
        "predicting_processing", "predicting_filling_gaps", "predicting_converting",
        "predicting_forecasting", "prediction_completed"
    ]
    if current_predict_status in busy_statuses:
         print(f"Prediction trigger ignored for session {session_id}, status: {current_predict_status}")
         raise HTTPException(
             status_code=status.HTTP_409_CONFLICT,
             detail=f"Prediction process is already '{current_predict_status}' for session {session_id}."
         )

    # --- Get necessary data from session (item name) ---
    item_name = session.get("coreTitle_insideQuotes") 
    if not item_name or item_name == "Not Determined":
        # Update status to error immediately if no title
        await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"prediction_status": "prediction_error", "prediction_error": "Missing item title for prediction"}}
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Item title not found in session. Cannot start prediction.")

    # --- Update Session Status (Mark as Processing/Starting) ---
    log.info(f"Triggering prediction for session {session_id}, item: '{item_name}'")
    await sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {
            "prediction_status": "processing", # General processing status
            # Or use the first specific step: "predicting_finding_id"
            "prediction_start_time": datetime.now(timezone.utc),
            "prediction_error": None # Clear previous errors
            }}
    )

    # --- Add Prediction Task to Background ---
    db_config = {"uri": config.DEFAULT_MONGO_URI, "db_name": config.APP_DB_NAME}
    background_tasks.add_task(
        run_prediction_and_store, # The async function doing the prediction work
        session_id=session_id,
        item_name=item_name,      # Pass the determined item name
        db_config=db_config
    )

    log.info(f"Background prediction task added for session {session_id}")
    return {"message": "Price prediction process initiated."}


@router.get("/predict-prices/status/{session_id}")
async def get_price_prediction_status(
    session_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    """
    Checks the status of the price prediction process and returns results if completed.
    """
    sessions_collection = request.app.sessions_collection

    # --- Fetch and Validate Session ---
    session = await get_session(session_id, sessions_collection)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")

    # --- Get Prediction Status and Data ---
    prediction_status = session.get("prediction_status", "not_started")

    def format_datetime(dt):
        if isinstance(dt, datetime):
            # Return in ISO 8601 format, which is standard for JSON APIs
            return dt.isoformat()
        return None # Return None if it's not a datetime object or is missing

    # --- Build response_data, formatting datetimes ---
    response_data = {
        "session_id": session_id,
        "prediction_status": prediction_status,
        "item_title": session.get("coreTitle_insideQuotes") or session.get("fullTitle", "N/A"),
        "start_time": format_datetime(session.get("prediction_start_time")), # Format start time
        "end_time": format_datetime(session.get("prediction_end_time"))      # Format end time
    }

    # Add results or error message based on status
    if prediction_status == "prediction_completed":
        response_data["results"] = session.get("prediction_results", [])
    elif prediction_status == "prediction_error":
        response_data["error_message"] = session.get("prediction_error", "Unknown prediction error")

    log.debug(f"Checked prediction status for session {session_id}: {prediction_status}")
    return JSONResponse(content=response_data)

# -------------------------------------------
# --- Notification Routes ---
# -------------------------------------------
@router.put("/notify-subscribe/{session_id}")
async def subscribe_to_notifications(
    session_id: str,
    subscription: SubscriptionRequest, # Send 'true' or 'false' in request body (e.g., {"subscribe": true})
    request: Request,
    current_user: Dict[str, Any] = Depends(get_user_from_auth_token)
):
    """
    Allows the user to subscribe/unsubscribe to price drop notifications for a session.

    """
    sessions_collection = request.app.sessions_collection
    subscribe = subscription.subscribe
    log.info(f"Subscription request for session {session_id}: Subscribe={subscribe}")

    # Fetch and Validate Session
    session = await get_session(session_id, sessions_collection)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Session access denied")

    # Optional: Check if scraping is completed before allowing subscription?
    if session.get("scrape_status") != "completed":
         log.warning(f"Attempt to subscribe for session {session_id} before scraping completed.")

    
    try:  # Update subscription status
        result = await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"notify_on_price_drop": bool(subscribe)}} # Ensure boolean storage
        )
        if result.matched_count == 0:
            # This shouldn't happen if get_session succeeded, but check anyway
            raise HTTPException(status_code=404, detail="Session not found during update.")

        action = "subscribed to" if subscribe else "unsubscribed from"
        log.info(f"User {current_user['email']} {action} price drop notifications for session {session_id}.")
        return {"message": f"Successfully {action} notifications."}

    except Exception as e:
        log.error(f"Failed to update notification status for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update notification preference.")



# -------------------------------------------
# --- Debugging Route (Optional) ---
# -------------------------------------------
# @router.post("/debug/trigger-price-check", status_code=200)
# async def debug_trigger_price_check(request: Request):
#     """
#     DEBUGGING ENDPOINT: Manually triggers the check_price_drops scheduled job.
#     Use with caution. Ensure DB config is available.
#     """
#     log.warning("!!! Received request to MANUALLY TRIGGER price drop check via DEBUG endpoint !!!")

#     # <<< EDIT: Simplified Check >>>
#     # Check if the function we want to call exists in the current global scope
#     if "check_price_drops" not in globals() or not callable(globals()["check_price_drops"]):
#         log.error("check_price_drops function not found or not callable for debug trigger.")
#         raise HTTPException(status_code=500, detail="Job function not ready.")
#     # We don't strictly need to check if 'scheduler' is initialized here,
#     # as we are calling the function directly, not through the scheduler instance.
#     # ------------------------------------

#     # Recreate DB config needed by the job
#     # Ensure config module is accessible here
#     try:
#         db_config_job = {"uri": config.DEFAULT_MONGO_URI, "db_name": config.APP_DB_NAME}
#         if not db_config_job["uri"] or not db_config_job["db_name"]:
#              raise ValueError("DB Config values missing.")
#     except (AttributeError, ValueError) as config_err:
#          log.error(f"Could not get DB config for debug trigger: {config_err}")
#          raise HTTPException(status_code=500, detail="Could not get DB configuration.")


#     # Manually run the job function (await it since it's async)
#     try:
#         log.info("Manually invoking check_price_drops...")
#         # <<< EDIT: Directly call the function from global scope >>>
#         await check_price_drops(db_config_job)
#         # -----------------------------------------------------
#         log.info("Manual invocation of check_price_drops finished.")
#         return {"message": "Manual price drop check triggered and completed."}
#     except Exception as e:
#         log.error(f"Error during manual invocation of check_price_drops: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error during manual trigger: {str(e)}")

# =============================================================================
# --- Include Router  ---
# =============================================================================
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for development...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)

# --- END OF FILE main.py ---
