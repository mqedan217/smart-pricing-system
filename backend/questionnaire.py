#--- START OF FILE questionnaire.py ---
import os
import json
import openai
import re
import logging
import sys # For basic console logging setup
from pymongo import MongoClient, errors as mongo_errors # Import pymongo
from datetime import datetime, timezone # For timestamping cache entries
from typing import List, Dict, Any, Optional # For type hinting
from motor.motor_asyncio import AsyncIOMotorCollection # USE ASYNC
# --- Import Configuration ---
try:
    import config # Import the config module
except ImportError:
    print("ERROR: config.py not found. Please ensure it exists in the same directory.")
    # Attempt to set some sane defaults so the script might partially run or show errors clearly
    class ConfigFallback:
        OPENAI_API_KEY = None
        DEFAULT_MONGO_URI = "mongodb://localhost:27017/" # Default URI
        CACHE_DB_NAME = "cache"
        CACHE_COLLECTION_NAME = "questionnaire_cache"
        OPENAI_QUESTION_MODEL="gpt-4o-mini"
        OPENAI_QUESTION_TEMP=0.3
        OPENAI_QUESTION_TIMEOUT=20.0
        OPENAI_TITLE_MODEL="gpt-4o-mini"
        OPENAI_TITLE_TEMP=0.3
        OPENAI_TITLE_TIMEOUT=20.0
        MAX_QUESTIONS=3
    config = ConfigFallback()
    print("Warning: Using fallback configuration values.")
# ---------------------------


# --- Basic Logging Setup (Get logger for this module) ---
# Assumes root logger is configured elsewhere (e.g., orchestrator)
log = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- OpenAI Client Initialization using config ---
client: Optional[openai.AsyncOpenAI] = None # Use type hint
if not config.OPENAI_API_KEY:
    log.error("OpenAI API key not found in configuration. AI features will fail.")
else:
    try:
        client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        log.info("Async OpenAI client initialized using config key.")
    except Exception as e:
        log.error(f"Failed to initialize Async OpenAI client: {e}", exc_info=True)
        client = None
# ------------------------------------------------
# --- Helper to create a consistent cache key ---
def create_cache_key(category: str, answers: list, cache_type: str) -> str:
    """Creates a unique document ID for the cache."""
    try:
        safe_category = re.sub(r'[^\w\-]+', '_', category)
        safe_answers_str = "_".join([re.sub(r'[^\w\-]+', '_', str(ans)) for ans in answers])
        # Use a separator that's unlikely to be in sanitized answers/category
        full_key = f"{cache_type}_{safe_category}_{safe_answers_str}"
        # Hash if too long? For now, just truncate. Be mindful of collisions if keys get very long.
        return full_key[:500] # Limit key length
    except Exception as e:
        log.error(f"Error creating cache key for {category}/{answers}: {e}")
        return f"ERROR_KEY_{datetime.now(timezone.utc).timestamp()}"


# --- Modified Cache Functions ---
async def load_cached_item(
        cache_collection : AsyncIOMotorCollection,
        category: str, answers: list, cache_type: str
        ) -> Optional[Any]:
    """Loads data from MongoDB cache asynchronously."""
    if cache_collection is None: return None # Silently return if no DB connection
    doc_id = create_cache_key(category, answers, cache_type)
    if "ERROR_KEY" in doc_id: return None
    try:
        cached_doc = await cache_collection.find_one({"_id": doc_id})
        if cached_doc and "cached_data" in cached_doc:
            return cached_doc["cached_data"]
        else:
            log.debug(f"Cache MISS in DB for key: ...{doc_id[-30:]}")
            return None
    except Exception as e:
        log.error(f"Unexpected error loading cache for key {doc_id}: {e}", exc_info=False)
        return None

async def save_item_to_cache(
        cache_collection:AsyncIOMotorCollection,
        category: str, answers: list, cache_type: str, data_to_cache: Any
        ) -> bool:
    """Saves data to MongoDB cache asynchronosly"""
    if cache_collection is None: return False # Silently return if no DB connection
    doc_id = create_cache_key(category, answers, cache_type)
    if "ERROR_KEY" in doc_id: return False
    cache_document = {
        "_id": doc_id, "category": category, "answers_key": answers,
        "cache_type": cache_type, "cached_data": data_to_cache,
        "created_at": datetime.now(timezone.utc)
    }
    try:
        result = await cache_collection.update_one(
            {"_id": doc_id}, {"$set": cache_document}, upsert=True
        )
        return True
    except Exception as e:
        log.error(f"Unexpected error saving cache for key {doc_id}: {e}", exc_info=False)
        return False

# --- Modified Core Functions to Use DB Cache & Config ---

async def generate_next_question(
        cache_collection:AsyncIOMotorCollection,
        category: str, answers: list
        ) -> dict:
    """Generates the next question, using MongoDB cache asynchronosly or OpenAI."""
    fallback_response = {"question": "[Error] Unable to generate question.", "choices": ["Retry", "Skip"]}
    if not client: log.error("OpenAI client unavailable."); return fallback_response

    current_answers = list(answers) if isinstance(answers, (list, tuple)) else []
    cached_question = await load_cached_item(cache_collection,category, current_answers, "question")
    if cached_question:
        if isinstance(cached_question, dict) and "question" in cached_question and "choices" in cached_question:
             return cached_question
        else: log.warning(f"Cached question from DB invalid format: {cached_question}")

    log.info(f"Generating new question (OpenAI) for {category}/{current_answers}")
    context = " > ".join(map(str, current_answers)) if current_answers else "None yet"
    # Keep the original prompt structure used previously
    user_message = f"""
Category: {category}
Previous answers: {context}

You are a product assistant helping a user identify an item in the above category.

Ask the next most helpful question to narrow it down, logically progressing from general to specific:
→ Brand or Subcategory → Model or style → Key Feature

Return valid JSON only.
"""
    try:
        response = await client.chat.completions.create(
            model=config.OPENAI_QUESTION_MODEL, # Use config
            messages=[
                {"role": "system", "content": "You ask 3 progressive filtering questions to identify any product across categories. Start general, then narrow by brand or type, then ask for specs or features. Return only structured JSON."},
                {"role": "user", "content": user_message.strip()}
            ],
            tools=[ { "type": "function", "function": { "name": "get_next_question", "description": "Ask the next narrowing product question based on the user's previous answers", "parameters": { "type": "object", "properties": { "question": {"type": "string"}, "choices": { "type": "array", "items": {"type": "string"} } }, "required": ["question", "choices"] } } } ],
            tool_choice={"type": "function", "function": {"name": "get_next_question"}},
            temperature=config.OPENAI_QUESTION_TEMP, # Use config
            timeout=config.OPENAI_QUESTION_TIMEOUT # Use config
        )
        # Robust access to response
        if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.tool_calls:
            log.error(f"Invalid OpenAI response structure (question): {response}")
            raise ValueError("Invalid OpenAI response structure")
        tool_call = response.choices[0].message.tool_calls[0]
        if not tool_call or tool_call.function.name != "get_next_question": raise ValueError("Incorrect/missing tool call")
        args = json.loads(tool_call.function.arguments)
        if "question" not in args or "choices" not in args: raise ValueError("Missing required keys in args")

        await save_item_to_cache(cache_collection, category, current_answers, "question", args)
        return args
    except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.APIError) as api_err:
         log.error(f"OpenAI API error (question): {api_err}", exc_info=False)
    except (json.JSONDecodeError, ValueError, AttributeError, IndexError, KeyError) as proc_err:
         log.error(f"Error processing OpenAI response (question): {proc_err}", exc_info=True)
    except Exception as e:
         log.error(f"Unexpected error during OpenAI call (question): {e}", exc_info=True)
    return fallback_response

async def identifyproduct(cache_collection: AsyncIOMotorCollection,category: str, answers: Any, specs: str, readyFromDatabase: bool = False) -> dict:
    """Generates/cleans product title using MongoDB cache or OpenAI with config values."""
    cache_key_answers = []
    cache_type = "core_title"
    cached_core_title = None
    saveToCache = False
    raw_input = ""
    job = ""

    # Determine if we should check cache
    if readyFromDatabase and isinstance(answers, str):
        log.debug("Skipping cache lookup/save for cleaning DB title + specs.")
    elif not readyFromDatabase and isinstance(answers, (list, tuple)):
        cache_key_answers = list(answers)
    else: log.debug(f"Unhandled input types for identifyproduct cache logic.")



    # Determine path: Use cache or generate
    if readyFromDatabase:
        saveToCache = False
        raw_input = f'"{answers}" {specs}'.strip()
        # Job description for cleaning DB title
        job = """
1. The core title (inside quotes): it is likely clean, leave as is or make minor corrections.
2. Clean the optional info (outside quotes): correct mistakes, simplify terms like "unused" → "new", remove redundancy.
3. Return the cleaned result in this exact format: "cleaned core title" cleaned extras
"""
    else:
        if cache_key_answers:
             cached_core_title = await load_cached_item(cache_collection,category, cache_key_answers, cache_type)

        if cached_core_title and isinstance(cached_core_title, str):
            log.info(f"Using cached core title '{cached_core_title}' from DB.")
            raw_input = f'"{cached_core_title}" {specs}'.strip()
            # Job description for cleaning cached core title
            job = """
1. The core title (inside quotes): it is likely clean, leave as is or make minor corrections.
2. Clean the optional info (outside quotes): correct mistakes, simplify terms, remove redundancy.
3. Return the cleaned result in this exact format: "cleaned core title" cleaned extras
"""
            saveToCache = False # Don't overwrite cache
        else:
            if cache_key_answers: log.info(f"Core title not cached for {cache_key_answers}. Generating.")
            answers_str = " > ".join(map(str, cache_key_answers))
            raw_input = f'"{answers_str}" {specs}'.strip()
             # Job description for generating title
            job = """
1. Clean the core title (inside quotes): remove filler words, preserve brand, model, key specs.
2. Clean the optional info (outside quotes): correct mistakes, simplify terms, remove redundancy.
3. Return the cleaned result in this exact format: "cleaned core title" cleaned extras
"""
            saveToCache = bool(cache_key_answers) # Only save if based on answers


    fallback_core = extract_inner_quotes(raw_input)
    fallback_result = {"product_title": raw_input, "core_title": fallback_core}

    if not client: log.error("OpenAI client unavailable."); return fallback_result

    # Prepare the full prompt
    prompt = f"""
You will receive a product title like this:
"core title inside quotes" optional info outside the quotes

Your job:{job}

🧪 Example:
Input: "apple iphone 13 storage capacity 128gb" color is blue and unused
Output: "apple iphone 13 128gb" blue new

Input: {raw_input}
"""
    try:
        response = await client.chat.completions.create(
             # Use title-specific config values if defined, else reuse question ones
            model=getattr(config, "OPENAI_TITLE_MODEL", config.OPENAI_QUESTION_MODEL),
            messages=[
                {"role": "system", "content": "You clean and reformat product titles and optimize them for SEO and web scarping."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=getattr(config, "OPENAI_TITLE_TEMP", config.OPENAI_QUESTION_TEMP),
            timeout=getattr(config, "OPENAI_TITLE_TIMEOUT", config.OPENAI_QUESTION_TIMEOUT)
        )

        # Refined Robust response processing
        if not response or not response.choices or not response.choices[0].message or response.choices[0].message.content is None:
            response_str = str(response) if response else "None"
            log.error(f"Invalid OpenAI response structure (title). Response: {response_str[:500]}...")
            raise ValueError("Invalid OpenAI response structure (missing choices/message/content)")

        cleaned = response.choices[0].message.content.strip()
        if cleaned.lower().startswith("output: "):
            cleaned = cleaned[len("Output: "):].strip()
            log.debug(f"Removed 'Output: ' prefix. Cleaned title: '{cleaned}'")
        core_title = extract_inner_quotes(cleaned)
        if not core_title: log.warning(f"OpenAI response missing quoted core title: '{cleaned}'")

        result = {"product_title": cleaned, "core_title": core_title if core_title else fallback_core }

        # Save the *extracted core title string* to cache if generated from answers
        if saveToCache and result["core_title"] and cache_key_answers:
           await save_item_to_cache(cache_collection,category, cache_key_answers, cache_type, result["core_title"])

        return result

    except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.APIError) as api_err:
         log.error(f"OpenAI API error (title): {api_err}", exc_info=False)
    except (json.JSONDecodeError, ValueError, AttributeError, IndexError, KeyError) as proc_err:
         log.error(f"Error processing OpenAI request/response (title): {proc_err}", exc_info=True)
    except Exception as e:
         log.error(f"Unexpected error during OpenAI call (title): {e}", exc_info=True)

    log.warning("Returning fallback title due to error during generation/cleaning.")
    return fallback_result # Return fallback on error

# --- extract_inner_quotes function ---
def extract_inner_quotes(text: str) -> str:
    """Safely extracts content inside the first pair of double quotes."""
    if not isinstance(text, str): return ""
    try:
        match = re.search(r'"(.*?)"', text)
        return match.group(1) if match else ""
    except Exception:
        log.warning(f"Regex error in extract_inner_quotes: {text[:50]}...")
        return ""
# --- END OF FILE questionnaire_final.py ---



# version for orchestrator 

# --- START OF FILE questionnaire_mongocache.py ---

# import os
# import json
# import openai
# import re
# import logging
# import sys # For basic console logging setup
# from pymongo import MongoClient, errors as mongo_errors # Import pymongo
# from datetime import datetime, timezone # For timestamping cache entries
# from typing import List, Dict, Any, Optional # For type hinting

# # --- Import Configuration ---
# try:
#     import config # Import the config module
# except ImportError:
#     print("ERROR: config.py not found. Please ensure it exists in the same directory.")
#     # Attempt to set some sane defaults so the script might partially run or show errors clearly
#     class ConfigFallback:
#         OPENAI_API_KEY = None
#         DEFAULT_MONGO_URI = "mongodb://localhost:27017/" # Default URI
#         CACHE_DB_NAME = "cache"
#         CACHE_COLLECTION_NAME = "questionnaire_cache"
#         OPENAI_QUESTION_MODEL="gpt-4o-mini"
#         OPENAI_QUESTION_TEMP=0.3
#         OPENAI_QUESTION_TIMEOUT=20.0
#         OPENAI_TITLE_MODEL="gpt-4o-mini"
#         OPENAI_TITLE_TEMP=0.3
#         OPENAI_TITLE_TIMEOUT=20.0
#         MAX_QUESTIONS=3
#     config = ConfigFallback()
#     print("Warning: Using fallback configuration values.")
# # ---------------------------


# # --- Basic Logging Setup (Get logger for this module) ---
# # Assumes root logger is configured elsewhere (e.g., orchestrator)
# log = logging.getLogger(__name__)
# # --- End Logging Setup ---

# # --- OpenAI Client Initialization using config ---
# client: Optional[openai.OpenAI] = None # Use type hint
# if not config.OPENAI_API_KEY:
#     log.error("OpenAI API key not found in configuration. AI features will fail.")
# else:
#     try:
#         client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
#         log.info("OpenAI client initialized using config key.")
#     except openai.AuthenticationError:
#         log.error("OpenAI authentication failed (using config key).", exc_info=True)
#         client = None
#     except Exception as e:
#         log.error(f"Failed to initialize OpenAI client using config key: {e}", exc_info=True)
#         client = None
# # ------------------------------------------------

# # --- MongoDB Cache Connection using config ---
# # Uses FEATURES_DB_URI from config by default, connects to specified Cache DB/Collection
# mongo_client_cache: Optional[MongoClient] = None
# cache_db = None
# cache_collection = None

# if config.DEFAULT_MONGO_URI: # Check if the URI was loaded successfully
#     try:
#         # Create a dedicated client for the cache for separation
#         mongo_client_cache = MongoClient(config.DEFAULT_MONGO_URI, serverSelectionTimeoutMS=3000, appName="QuestionnaireCache")
#         mongo_client_cache.admin.command('ping') # Verify connection
#         # Use cache DB and Collection names from config
#         cache_db = mongo_client_cache[config.CACHE_DB_NAME]
#         cache_collection = cache_db[config.CACHE_COLLECTION_NAME]
#         log.info(f"MongoDB connection for cache OK (DB: {config.CACHE_DB_NAME}, Coll: {config.CACHE_COLLECTION_NAME}).")
#         # Optional: Consider creating TTL index here if needed (uncomment below)
#         # try:
#         #     cache_collection.create_index("created_at", expireAfterSeconds=30*24*60*60) # 30 days TTL
#         #     log.info("Ensured TTL index exists on cache collection.")
#         # except mongo_errors.OperationFailure:
#         #     log.debug("TTL index likely already exists.") # Ignore if index exists
#         # except Exception as idx_e:
#         #     log.warning(f"Could not ensure TTL index on cache collection: {idx_e}")
#     except mongo_errors.ConnectionFailure as e:
#         log.error(f"MongoDB connection for cache failed using URI from config: {e}. Caching disabled.", exc_info=False)
#     except Exception as e:
#         log.error(f"Error setting up dedicated cache DB connection: {e}", exc_info=True)
# else:
#     log.error("Cannot connect to Cache DB: FEATURES_DB_URI not found in configuration.")
# # -------------------------------------------------------------------


# # --- Helper to create a consistent cache key ---
# def create_cache_key(category: str, answers: list, cache_type: str) -> str:
#     """Creates a unique document ID for the cache."""
#     try:
#         safe_category = re.sub(r'[^\w\-]+', '_', category)
#         safe_answers_str = "_".join([re.sub(r'[^\w\-]+', '_', str(ans)) for ans in answers])
#         # Use a separator that's unlikely to be in sanitized answers/category
#         full_key = f"{cache_type}_{safe_category}_{safe_answers_str}"
#         # Hash if too long? For now, just truncate. Be mindful of collisions if keys get very long.
#         return full_key[:500] # Limit key length
#     except Exception as e:
#         log.error(f"Error creating cache key for {category}/{answers}: {e}")
#         return f"ERROR_KEY_{datetime.now(timezone.utc).timestamp()}"


# # --- Modified Cache Functions ---
# def load_cached_item(category: str, answers: list, cache_type: str) -> Optional[Any]:
#     """Loads data from MongoDB cache, returning None on any error or miss."""
#     if cache_collection is None: return None # Silently return if no DB connection
#     doc_id = create_cache_key(category, answers, cache_type)
#     if "ERROR_KEY" in doc_id: return None
#     try:
#         cached_doc = cache_collection.find_one({"_id": doc_id})
#         if cached_doc and "cached_data" in cached_doc:
#             log.info(f"Cache HIT from DB for type '{cache_type}', key: ...{doc_id[-30:]}")
#             return cached_doc["cached_data"]
#         else:
#             log.debug(f"Cache MISS in DB for key: ...{doc_id[-30:]}")
#             return None
#     except mongo_errors.PyMongoError as e:
#         log.error(f"MongoDB error loading cache for key {doc_id}: {e}", exc_info=False)
#         return None
#     except Exception as e:
#         log.error(f"Unexpected error loading cache for key {doc_id}: {e}", exc_info=True)
#         return None

# def save_item_to_cache(category: str, answers: list, cache_type: str, data_to_cache: Any) -> bool:
#     """Saves data to MongoDB cache, logging errors."""
#     if cache_collection is None: return False # Silently return if no DB connection
#     doc_id = create_cache_key(category, answers, cache_type)
#     if "ERROR_KEY" in doc_id: return False
#     cache_document = {
#         "_id": doc_id, "category": category, "answers_key": answers,
#         "cache_type": cache_type, "cached_data": data_to_cache,
#         "created_at": datetime.now(timezone.utc)
#     }
#     try:
#         result = cache_collection.update_one(
#             {"_id": doc_id}, {"$set": cache_document}, upsert=True
#         )
#         log.info(f"Saved '{cache_type}' to DB cache. Key: ...{doc_id[-30:]}. Upserted: {result.upserted_id is not None}, Mod: {result.modified_count > 0}")
#         return True
#     except mongo_errors.PyMongoError as e:
#         log.error(f"MongoDB error saving cache for key {doc_id}: {e}", exc_info=False)
#         return False
#     except Exception as e:
#         log.error(f"Unexpected error saving cache for key {doc_id}: {e}", exc_info=True)
#         return False

# # --- Modified Core Functions to Use DB Cache & Config ---

# def generate_next_question(category: str, answers: list) -> dict:
#     """Generates the next question, using MongoDB cache or OpenAI with config values."""
#     fallback_response = {"question": "[Error] Unable to generate question.", "choices": ["Retry", "Skip"]}
#     if not client: log.error("OpenAI client unavailable."); return fallback_response

#     current_answers = list(answers) if isinstance(answers, (list, tuple)) else []
#     cached_question = load_cached_item(category, current_answers, "question")
#     if cached_question:
#         if isinstance(cached_question, dict) and "question" in cached_question and "choices" in cached_question:
#              return cached_question
#         else: log.warning(f"Cached question from DB invalid format: {cached_question}")

#     log.info(f"Generating new question (OpenAI) for {category}/{current_answers}")
#     context = " > ".join(map(str, current_answers)) if current_answers else "None yet"
#     # Keep the original prompt structure used previously
#     user_message = f"""
# Category: {category}
# Previous answers: {context}

# You are a product assistant helping a user identify an item in the above category.

# Ask the next most helpful question to narrow it down, logically progressing from general to specific:
# → Brand or Subcategory → Model or style → Key Feature

# Return valid JSON only.
# """
#     try:
#         response = client.chat.completions.create(
#             model=config.OPENAI_QUESTION_MODEL, # Use config
#             messages=[
#                 {"role": "system", "content": "You ask 3 progressive filtering questions to identify any product across categories. Start general, then narrow by brand or type, then ask for specs or features. Return only structured JSON."},
#                 {"role": "user", "content": user_message.strip()}
#             ],
#             tools=[ { "type": "function", "function": { "name": "get_next_question", "description": "Ask the next narrowing product question based on the user's previous answers", "parameters": { "type": "object", "properties": { "question": {"type": "string"}, "choices": { "type": "array", "items": {"type": "string"} } }, "required": ["question", "choices"] } } } ],
#             tool_choice={"type": "function", "function": {"name": "get_next_question"}},
#             temperature=config.OPENAI_QUESTION_TEMP, # Use config
#             timeout=config.OPENAI_QUESTION_TIMEOUT # Use config
#         )
#         # Robust access to response
#         if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.tool_calls:
#             log.error(f"Invalid OpenAI response structure (question): {response}")
#             raise ValueError("Invalid OpenAI response structure")
#         tool_call = response.choices[0].message.tool_calls[0]
#         if not tool_call or tool_call.function.name != "get_next_question": raise ValueError("Incorrect/missing tool call")
#         args = json.loads(tool_call.function.arguments)
#         if "question" not in args or "choices" not in args: raise ValueError("Missing required keys in args")

#         save_item_to_cache(category, current_answers, "question", args)
#         return args
#     except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.APIError) as api_err:
#          log.error(f"OpenAI API error (question): {api_err}", exc_info=False)
#     except (json.JSONDecodeError, ValueError, AttributeError, IndexError, KeyError) as proc_err:
#          log.error(f"Error processing OpenAI response (question): {proc_err}", exc_info=True)
#     except Exception as e:
#          log.error(f"Unexpected error during OpenAI call (question): {e}", exc_info=True)
#     return fallback_response


# def identifyproduct(category: str, answers: Any, specs: str, readyFromDatabase: bool = False) -> dict:
#     """Generates/cleans product title using MongoDB cache or OpenAI with config values."""
#     cache_key_answers = []
#     cache_type = "core_title"

#     # Determine if we should check cache
#     if readyFromDatabase and isinstance(answers, str):
#         log.debug("Skipping cache lookup/save for cleaning DB title + specs.")
#     elif not readyFromDatabase and isinstance(answers, (list, tuple)):
#         cache_key_answers = list(answers)
#     else: log.debug(f"Unhandled input types for identifyproduct cache logic.")

#     cached_core_title = None
#     saveToCache = False
#     raw_input = ""
#     job = ""

#     # Determine path: Use cache or generate
#     if readyFromDatabase:
#         saveToCache = False
#         raw_input = f'"{answers}" {specs}'.strip()
#         # Job description for cleaning DB title
#         job = """
# 1. The core title (inside quotes): it is likely clean, leave as is or make minor corrections.
# 2. Clean the optional info (outside quotes): correct mistakes, simplify terms like "unused" → "new", remove redundancy.
# 3. Return the cleaned result in this exact format: "cleaned core title" cleaned extras
# """
#     else:
#         if cache_key_answers:
#              cached_core_title = load_cached_item(category, cache_key_answers, cache_type)

#         if cached_core_title and isinstance(cached_core_title, str):
#             log.info(f"Using cached core title '{cached_core_title}' from DB.")
#             raw_input = f'"{cached_core_title}" {specs}'.strip()
#             # Job description for cleaning cached core title
#             job = """
# 1. The core title (inside quotes): it is likely clean, leave as is or make minor corrections.
# 2. Clean the optional info (outside quotes): correct mistakes, simplify terms, remove redundancy.
# 3. Return the cleaned result in this exact format: "cleaned core title" cleaned extras
# """
#             saveToCache = False # Don't overwrite cache
#         else:
#             if cache_key_answers: log.info(f"Core title not cached for {cache_key_answers}. Generating.")
#             answers_str = " > ".join(map(str, cache_key_answers))
#             raw_input = f'"{answers_str}" {specs}'.strip()
#              # Job description for generating title
#             job = """
# 1. Clean the core title (inside quotes): remove filler words, preserve brand, model, key specs.
# 2. Clean the optional info (outside quotes): correct mistakes, simplify terms, remove redundancy.
# 3. Return the cleaned result in this exact format: "cleaned core title" cleaned extras
# """
#             saveToCache = bool(cache_key_answers) # Only save if based on answers


#     fallback_core = extract_inner_quotes(raw_input)
#     fallback_result = {"product_title": raw_input, "core_title": fallback_core}

#     if not client: log.error("OpenAI client unavailable."); return fallback_result

#     # Prepare the full prompt
#     prompt = f"""
# You will receive a product title like this:
# "core title inside quotes" optional info outside the quotes

# Your job:{job}

# 🧪 Example:
# Input: "apple iphone 13 storage capacity 128gb" color is blue and unused
# Output: "apple iphone 13 128gb" blue new

# Input: {raw_input}
# """
#     # log.debug(f"IdentifyProduct Prompt:\n{prompt.strip()}") # Uncomment for debugging

#     try:
#         response = client.chat.completions.create(
#              # Use title-specific config values if defined, else reuse question ones
#             model=getattr(config, "OPENAI_TITLE_MODEL", config.OPENAI_QUESTION_MODEL),
#             messages=[
#                 {"role": "system", "content": "You clean and reformat product titles and optimize them for SEO and web scarping."},
#                 {"role": "user", "content": prompt.strip()}
#             ],
#             temperature=getattr(config, "OPENAI_TITLE_TEMP", config.OPENAI_QUESTION_TEMP),
#             timeout=getattr(config, "OPENAI_TITLE_TIMEOUT", config.OPENAI_QUESTION_TIMEOUT)
#         )

#         # Refined Robust response processing
#         if not response or not response.choices or not response.choices[0].message or response.choices[0].message.content is None:
#             response_str = str(response) if response else "None"
#             log.error(f"Invalid OpenAI response structure (title). Response: {response_str[:500]}...")
#             raise ValueError("Invalid OpenAI response structure (missing choices/message/content)")

#         cleaned = response.choices[0].message.content.strip()
#         core_title = extract_inner_quotes(cleaned)
#         if not core_title: log.warning(f"OpenAI response missing quoted core title: '{cleaned}'")

#         result = {"product_title": cleaned, "core_title": core_title if core_title else fallback_core }

#         # Save the *extracted core title string* to cache if generated from answers
#         if saveToCache and result["core_title"] and cache_key_answers:
#             save_item_to_cache(category, cache_key_answers, cache_type, result["core_title"])

#         return result

#     except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.APIError) as api_err:
#          log.error(f"OpenAI API error (title): {api_err}", exc_info=False)
#     except (json.JSONDecodeError, ValueError, AttributeError, IndexError, KeyError) as proc_err:
#          log.error(f"Error processing OpenAI request/response (title): {proc_err}", exc_info=True)
#     except Exception as e:
#          log.error(f"Unexpected error during OpenAI call (title): {e}", exc_info=True)

#     log.warning("Returning fallback title due to error during generation/cleaning.")
#     return fallback_result # Return fallback on error


# # --- extract_inner_quotes function ---
# def extract_inner_quotes(text: str) -> str:
#     """Safely extracts content inside the first pair of double quotes."""
#     if not isinstance(text, str): return ""
#     try:
#         match = re.search(r'"(.*?)"', text)
#         return match.group(1) if match else ""
#     except Exception:
#         log.warning(f"Regex error in extract_inner_quotes: {text[:50]}...")
#         return ""

# # # --- END OF FILE questionnaire_final.py ---