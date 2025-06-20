# --- START OF FILE data_processing.py ---
import httpx
import json
import re
import numpy as np
import logging
import faiss
import torch

# Import necessary libraries from original code
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from scipy.stats import iqr # Import iqr specifically
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import errors as mongo_errors
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
import sys

# --- Import Configuration ---
try:
    import config # Import the config module
except ImportError:
    print("ERROR: config.py not found. Please ensure it exists.")
    # Define fallback defaults if config cannot be imported
    class ConfigFallback:
        RESNET_WEIGHTS_STR="DEFAULT" # Default weight string for ResNet50
        AED_TO_USD=0.27; JOD_TO_USD=1.41
        W2V_PARAMS={"vector_size": 1024, "window": 3, "min_count": 1, "workers": -1} # Default W2V params
        RELEVANCE_MAX_RESULTS=8 # Default limit for main products
        IQR_MULTIPLIER=1.0; # Default limit for related
    config = ConfigFallback()
    print("Warning: Using fallback configuration values for data_processing.")
# ---------------------------


# --- Logging Setup ---
log = logging.getLogger(__name__)
# ---------------------

# --- ResNet50 Model Loading using config ---
model: Optional[torch.nn.Module] = None
transform: Optional[transforms.Compose] = None
RESNET_LOADED_DATA_PROCESSING: bool = False
try:
    weights_enum_name = config.RESNET_WEIGHTS_STR.upper()
    weights_enum = getattr(ResNet50_Weights, weights_enum_name, ResNet50_Weights.DEFAULT)
    if not isinstance(weights_enum, ResNet50_Weights):
        log.warning(f"Invalid ResNet weight '{config.RESNET_WEIGHTS_STR}' in config. Using DEFAULT.")
        weights_enum = ResNet50_Weights.DEFAULT
    log.info(f"Loading ResNet50 model for data_processing with weights: {weights_enum.name}")
    model = resnet50(weights=weights_enum)
    model.eval(); model = torch.nn.Sequential(*list(model.children())[:-1])
    transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    RESNET_LOADED_DATA_PROCESSING = True
    log.info("ResNet50 model/transform loaded successfully for data_procesing.")
except Exception as e: log.error(f"Failed to load ResNet50 model in data_processing: {e}", exc_info=True)
# ----------------------------------------------------------------

# --- Configuration Values ---
AED_to_USD: float = config.AED_TO_USD
JOD_to_USD: float = config.JOD_TO_USD
DEFAULT_CURRENCY: str = "USD"

# --- Helper Functions (Original Logic with Error Handling) ---

def convert_price(price_str: Optional[str]) -> Optional[float]:
    if not isinstance(price_str, str) or not price_str.strip(): return None
    try:
        price_str_cleaned = price_str.replace(",", "").split("+", 1)[0].strip()
        if "AED" in price_str: return round(float(price_str_cleaned.replace("AED", "").strip()) * AED_to_USD, 2)
        elif "JOD" in price_str: return round(float(price_str_cleaned.replace("JOD", "").strip()) * JOD_to_USD, 2)
        elif "دينار" in price_str: return round(float(price_str_cleaned.replace("دينار", "").strip()) * JOD_to_USD, 2)
        elif "$" in price_str: return round(float(price_str_cleaned.replace("$", "").strip()), 2)
        else: return None # Original implicit behavior
    except (ValueError, TypeError): return None # Return None on error

def clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Original clean_data logic from first provided file."""
    if not isinstance(data, dict): log.error("clean_data input invalid."); return data
    cleaned_data = {k: [i.copy() if isinstance(i, dict) else i for i in v] if isinstance(v, list) else v for k, v in data.items()}
    site_keys = ['amazon_products', 'ebay_products', 'google_shopping', 'opensooq_products'] # Include Google as per later version
    for site_key in site_keys:
        products_in = cleaned_data.get(site_key)
        if isinstance(products_in, list):
            valid_products = []
            for product in products_in:
                if not isinstance(product, dict): continue
                price_str = product.get('price')
                # Original filter: Skip N/A strings, not None check? Be explicit.
                if price_str is None or (isinstance(price_str, str) and "N/A" in price_str): continue

                # Original eBay range logic
                if site_key == 'ebay_products' and isinstance(price_str, str) and 'to' in price_str.lower():
                    try:
                        p_range = price_str.split('to')
                        p1 = convert_price(p_range[0]); p2 = convert_price(p_range[1])
                        # Original code used float() directly on the result of convert_price, we keep that intent
                        # Using round for consistency as added previously
                        product['price_usd'] = round((p1 + p2) / 2, 2) if p1 is not None and p2 is not None else None
                    except: product['price_usd'] = None # Original broad except
                else: product['price_usd'] = convert_price(price_str) # Uses config rates implicitly

                # Original Keep condition: price_usd converted AND title exists and != "N/A"
                if product.get('price_usd') is not None and product.get('title') and product['title'] != "N/A":
                    valid_products.append(product)
            cleaned_data[site_key] = valid_products
            log.debug(f"Cleaned {site_key}: {len(valid_products)} valid items.")
        else:
            if site_key in data: log.warning(f"clean_data: '{site_key}' key present but not a list.")
            cleaned_data[site_key] = []
    return cleaned_data


def adjust_search_tokens(search_term: str) -> List[str]:
    """Original adjust_search_tokens logic."""
    if not isinstance(search_term, str): return []
    try:
        inside = re.findall(r'"(.*?)"', search_term)
        outside = re.sub(r'"(.*?)"', '', search_term).strip().split()
        adjusted = []
        for i, phrase in enumerate(inside):
            tokens = phrase.lower().split(); repeat = max(1, 20 - i * 2); adjusted.extend(tokens * repeat)
        for word in outside: adjusted.append(word.lower())
        return adjusted
    except Exception as e: log.error(f"Adjust tokens error: {e}"); return search_term.lower().split()

# Original run_word2vec (returns sorted list) using config params
def run_word2vec(products: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
    """Original logic: W2V -> FAISS -> Filter(score>0) -> Sort. Uses config."""
    if not products: return []
    try:
        search_tokens = adjust_search_tokens(search_term)
        tokenized_titles = [str(p.get('title', '')).lower().split() for p in products]
        valid_indices = [i for i, t in enumerate(tokenized_titles) if t]
        valid_tokenized_titles = [tokenized_titles[i] for i in valid_indices]
        if not valid_tokenized_titles or not search_tokens: log.warning("W2V: No tokens."); return products

        corpus = valid_tokenized_titles + [search_tokens]
        w2v_params = config.W2V_PARAMS
        model_w2v = Word2Vec(sentences=corpus, **w2v_params) # Pass params dict
        dim = model_w2v.vector_size

        def avg_vector(tokens): vecs=[model_w2v.wv[w] for w in tokens if w in model_w2v.wv]; return np.mean(vecs, axis=0) if vecs else np.zeros(dim)
        search_vec = avg_vector(search_tokens)
        title_vecs_list = [avg_vector(t) for t in valid_tokenized_titles]
        title_vecs_np = np.zeros((len(products), dim), dtype='float32')
        for i, vec in enumerate(title_vecs_list): title_vecs_np[valid_indices[i]] = vec

        if title_vecs_np.shape[0] == 0: log.warning("FAISS: No vectors."); return products

        index = faiss.IndexFlatL2(dim); index.add(title_vecs_np)
        search_vec_np = search_vec.reshape(1, -1).astype('float32')
        k = min(10, index.ntotal); # Original k=10
        if k == 0: log.warning("FAISS: Index empty."); return products
        distances, indices = index.search(search_vec_np, k)

        for p in products: p['relevance_score'] = 0.0 # Initialize
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(products): products[idx]['relevance_score'] = round(1/(distances[0][i]+1e-10), 4)

        # Original filter and sort
        sorted_prods = sorted(products, key=lambda x: x['relevance_score'], reverse=True)
        log.debug(f"W2V returning {len(sorted_prods)} sorted products.")
        return sorted_prods # Original function returned only one list

    except Exception as e: log.error(f"W2V/FAISS error: {e}", exc_info=True)
    for p in products: p['relevance_score'] = 0.0
    return products # Return original list on error


# Original apply_quartile_weights
def apply_quartile_weights(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not products: return []
    n = len(products)
    if n == 0: return []
    try:
        weights = np.linspace(4, 1, 4)
        for i, p in enumerate(products): p['quartile_weight'] = weights[min(int(i / (n * 0.25)), 3)]
        return products
    except Exception as e: log.error(f"Weighting error: {e}");
    for p in products: p['quartile_weight'] = 1.0; return products

# Original expand_by_weight
def expand_by_weight(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not products: return []
    expanded = []
    try:
        for p in products: weight = p.get('quartile_weight', 1); repeat = max(1, int(weight)); expanded.extend([p] * repeat)
        return expanded
    except Exception as e: log.error(f"Weight expansion error: {e}"); return products


# Original exclude_anomalies_iqr using config multiplier
def exclude_anomalies_iqr(products: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Original logic, uses config IQR multiplier."""
    if not products: return [], []
    prices = [p.get('price_usd') for p in products if isinstance(p.get('price_usd'), (int, float))]
    if len(prices) < 2: return products, []
    try:
        Q1, Q3 = np.percentile(prices, [25, 75]); IQR_val = Q3 - Q1
        multiplier = config.IQR_MULTIPLIER # Use config multiplier
        lower = Q1 - multiplier * IQR_val; upper = Q3 + multiplier * IQR_val
        filtered, removed = [], []
        for p in products:
            price = p.get('price_usd')
            if isinstance(price, (int, float)) and lower <= price <= upper: filtered.append(p)
            else: removed.append(p)
        log.debug(f"IQR ({multiplier}x): Kept {len(filtered)}, Removed {len(removed)}.")
        return filtered, removed
    except Exception as e: log.error(f"IQR error: {e}"); return products, []

# Original calculate_weighted_avg_price
def calculate_weighted_avg_price(products: List[Dict[str, Any]]) -> float:
    if not products: return 0.0
    ws, w_sum = 0.0, 0.0; count = 0
    try:
        for p in products:
            price = p.get('price_usd'); relevance = p.get('relevance_score')
            if isinstance(price, (int, float)) and isinstance(relevance, (int, float)):
                ws += price * relevance; w_sum += relevance; count += 1
        if w_sum == 0: # Fallback
            prices = [p.get('price_usd') for p in products if isinstance(p.get('price_usd'),(int, float))];
            return round(np.mean(prices), 2) if prices else 0.0
        avg = ws / w_sum; log.debug(f"Weighted avg: {avg:.2f}"); return round(avg, 2)
    except Exception as e: log.error(f"Weighted avg error: {e}"); return 0.0

# Original calculate_price_stats
def calculate_price_stats(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {'min_price': None, 'max_price': None, 'avg_price': 0.0, 'min_product': None, 'max_product': None}
    if not products: return stats
    valid = [p for p in products if isinstance(p.get('price_usd'), (int, float))]
    if not valid: return stats
    try:
        min_p = min(valid, key=lambda x: x['price_usd']); max_p = max(valid, key=lambda x: x['price_usd'])
        avg_p = calculate_weighted_avg_price(valid)
        stats.update({ 'min_price': min_p['price_usd'], 'max_price': max_p['price_usd'],
                       'avg_price': avg_p, 'min_product': min_p, 'max_product': max_p })
    except Exception as e: log.error(f"Price stats error: {e}"); stats['avg_price'] = 0.0
    return stats

# Original calculate_overall_stats (NO Google)
def calculate_overall_stats(amazon_products: List, ebay_products: List, google_products: List, opensooq_products: List) -> Dict[str, Any]: # Removed google_products arg
    """Original overall stats calc (NO Google)."""
    all_products = []
    if isinstance(amazon_products, list): all_products.extend(amazon_products)
    if isinstance(ebay_products, list): all_products.extend(ebay_products)
    if isinstance(google_products, list): all_products.extend(google_products) # Google included in later version
    if isinstance(opensooq_products, list): all_products.extend(opensooq_products)
    if not all_products:
        log.warning("Cannot calculate overall stats: Combined list empty.")
        return {'min_price': None, 'max_price': None, 'avg_price': 0.0, 'min_product': None, 'max_product': None}
    return calculate_price_stats(all_products) # Uses weighted average

# Original eliminate_similar_products
def eliminate_similar_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not products: return []
    unique = []; seen = set()
    for p in products:
        try: site=p.get('site'); rank=p.get('rank'); key=(site, rank)
        except: continue # Skip if error getting key
        if key not in seen: unique.append(p); seen.add(key)
    log.debug(f"Dedupe (site, rank) kept {len(unique)} items.")
    return unique

# Original remove_weight_and_relevance
def remove_weight_and_relevance(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not products: return []
    cleaned = []
    keys = ['quartile_weight', 'relevance_score']
    for p in products:
        if not isinstance(p, dict): continue
        cp = p.copy()
        for k in keys:
            if k in cp: del cp[k] # Simpler deletion
        cleaned.append(cp)
    return cleaned

# Original finalresults sequence using config values, including Google
def finalresults(data: Dict[str, Any], search_term: str) -> Dict[str, Any]:
    """Processes scraped data using original sequence, includes Google, uses config."""
    log.info(f"Starting finalresults processing (User's Filter Logic + Google + config) for search: '{search_term}'")
    # Default output structure
    final_output = {k: {"products": [], "stats": {}} for k in ["amazon", "ebay", "google_shopping", "opensooq"]}
    final_output["overall_stats"] = {}
    final_output["related_products"] = []
    # Default intermediate structures
    initial = {'amazon': [], 'ebay': [], 'google': [], 'opensooq': []} # Original data
    all_scored = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]} # Results from run_word2vec
    sorted_main = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]} # Top N after filtering limit
    related_candidates = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]} # Items after filter limit
    weighted = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]}
    expanded = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]}
    final_iqr = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]}
    removed_by_iqr = {'amazon': [], 'ebay': [], 'google': [], 'opensooq':[]}
    site_keys = ['amazon', 'ebay', 'google', 'opensooq'] # Original keys

    try:
        # 1. Clean Data
        cleaned_data = clean_data(data) # Uses config rates
        initial['amazon'] = cleaned_data.get("amazon_products", [])
        initial['ebay'] = cleaned_data.get("ebay_products", [])
        initial['google'] = cleaned_data.get("google_shopping", [])
        initial['opensooq'] = cleaned_data.get("opensooq_products", []) # Original logic

        # 2. Run Word2Vec (returns FULL sorted list)
        log.info("Calculating relevance scores (original params from config)...")
        try:
            tasks, keys_for_parallel = [], []
            w2v_workers = config.W2V_PARAMS.get("workers", 4)
            for key in site_keys:
                if initial[key]: tasks.append(delayed(run_word2vec)(initial[key], search_term)); keys_for_parallel.append(key)

            if tasks:
                 scored_results = Parallel(n_jobs=w2v_workers)(tasks)
                 result_map = dict(zip(keys_for_parallel, scored_results))
                 for key in site_keys: all_scored[key] = result_map.get(key, [])
            else: log.warning("No products to score.")

        except Exception as parallel_err:
            log.error(f"Parallel W2V Error: {parallel_err}. Running seq.", exc_info=True)
            for key in site_keys: all_scored[key] = run_word2vec(initial[key], search_term) if initial[key] else []

        # --- 3 & 4. Filter by Relevance Limit & Extract related products ---
        filter_limit = config.RELEVANCE_MAX_RESULTS # Get limit from config (e.g., 5 or 10)
        log.info(f"Applying relevance filter limit: Taking top {filter_limit} items per site.")
        for key in site_keys:
            if len(all_scored[key]) > filter_limit:
                log.debug(f"Filtering {key}: Original={len(all_scored[key])}, Filtered={filter_limit}.")
                sorted_main[key] = all_scored[key][:filter_limit]
                related_candidates[key] = all_scored[key][filter_limit:] # Items after the limit
            else:
                log.debug(f"Filtering {key}: Original={len(all_scored[key])}, Filtered={len(all_scored[key])}.")
                sorted_main[key] = all_scored[key] # Keep all if less than limit
                related_candidates[key] = [] # No related candidates if all are kept
        # --- End Filter ---


        # 5. Apply Quartile Weights (Applied to the filtered 'sorted_main' lists)
        for key in site_keys: weighted[key] = apply_quartile_weights(sorted_main[key])

        # 6. Expand by Weights (Applied to weighted lists)
        for key in site_keys: expanded[key] = expand_by_weight(weighted[key])

        # 7. Exclude Anomalies (Applied to expanded lists, uses config multiplier)
        for key in site_keys: final_iqr[key], removed_by_iqr[key] = exclude_anomalies_iqr(expanded[key])

        # 8. Calculate Price Stats (Based on IQR filtered expanded lists)
        for key_out, key_in in zip(["amazon", "ebay", "google_shopping", "opensooq"], site_keys):
            final_output[key_out]["stats"] = calculate_price_stats(final_iqr[key_in])

        # 9. Calculate Overall Stats (Includes Google as requested previously)
        final_output["overall_stats"] = calculate_overall_stats(
            final_iqr['amazon'], final_iqr['ebay'], final_iqr['google'], final_iqr['opensooq']
        )

        # 10. Final Unique Product Lists (Original Dedupe applied to IQR filtered expanded lists)
        amazon_unique = eliminate_similar_products(final_iqr['amazon'])
        ebay_unique = eliminate_similar_products(final_iqr['ebay'])
        google_unique = eliminate_similar_products(final_iqr['google'])
        opensooq_unique = eliminate_similar_products(final_iqr['opensooq']) # Original logic

        # 11. Remove Temporary Fields
        final_output["amazon"]["products"] = remove_weight_and_relevance(amazon_unique)
        final_output["ebay"]["products"] = remove_weight_and_relevance(ebay_unique)
        final_output["google_shopping"]["products"] = remove_weight_and_relevance(google_unique)
        final_output["opensooq"]["products"] = remove_weight_and_relevance(opensooq_unique) # Original logic

        # 12. Process RELATED Products
        # Combine IQR removals (from main processing) + the initially separated related candidates
        related_products_raw = (removed_by_iqr['amazon'] + removed_by_iqr['ebay'] + removed_by_iqr['google'] + removed_by_iqr['opensooq'] +
                                related_candidates['amazon'] + related_candidates['ebay'] + related_candidates['google'] + related_candidates['opensooq'])
        related_cleaned = eliminate_similar_products(related_products_raw) # Original dedupe
        final_output["related_products"] = remove_weight_and_relevance(related_cleaned)
        # Use limit from config
        final_output["related_products"] = final_output["related_products"]

        log.info("Finalresults processing complete (User Filter Logic + Google + config).")

    except Exception as e:
        log.error(f"Critical error during finalresults: {e}.", exc_info=True)
        # Fallback structure
        for k_out in final_output:
            if isinstance(final_output[k_out], dict): final_output[k_out] = {"products":[], "stats":{}}
            elif k_out == "related_products": final_output[k_out] = []
            elif k_out == "overall_stats": final_output[k_out] = {}
    return final_output

# --- Feature Extraction and Storage using config ---

async def extract_vector_from_url(url: str) -> Optional[np.ndarray]:
    """Extracts feature vector using config model/transform."""
    if not RESNET_LOADED_DATA_PROCESSING or model is None or transform is None:
        log.error("ResNet not loaded via config."); return None
    if not url or not isinstance(url, str) or not url.startswith(('http:','https:')):
        log.debug(f"Invalid URL for vector extraction: {url}")
        return None
    try:
        headers={'User-Agent':'Mozilla/5.0'}
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response= await client.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            ct=response.headers.get('Content-Type','').lower()
            if not ct.startswith('image/'):
                log.warning(f"URL not image (Type:{ct}): {url}")
                return None
            image_bytes = await response.aread()
            if not image_bytes: 
                log.warning(f"Received empty content from image URL: {url}")
                return None
            img=Image.open(BytesIO(response.content)).convert("RGB")
            tensor=transform(img).unsqueeze(0)
            with torch.no_grad():
                vec=model(tensor).squeeze().numpy()
            return vec.astype('float32')
    except httpx.HTTPStatusError as e:
        log.warning(f"HTTP error {e.response.status_code} extracting vector from {url}: {e}")
        return None
    except httpx.RequestError as e:
        # This catches connection errors, timeouts, etc.
        log.warning(f"Network error extracting vector from {url}: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during image processing or model inference
        log.warning(f"Unexpected error extracting vector from {url}: {e}", exc_info=False) # Keep logs concise
        return None

# --- appendToDoc using config ---
async def appendToDoc(category: str, products: List[Dict], coreTitle: str, features_coll: AsyncIOMotorCollection):
    """Appends product features to MongoDB, uses passed collection."""
    if features_coll is None:
         log.error("DB collection missing for appendToDoc."); return 0
    if not products: 
        return 0
    if not RESNET_LOADED_DATA_PROCESSING:
        log.error(f"Cannot append features: ResNet not loaded."); return 0

    added_count = 0; processed_count = 0; site = "unknown"
    for p in products:
        if not isinstance(p, dict): continue
        processed_count += 1
        title = coreTitle if coreTitle else p.get('title', 'Unknown')
        img_url = p.get("image_url", "N/A"); site = p.get("site", site); site = site.split("-")[0] # Get base site name
        if not img_url or img_url == "N/A": continue
        vector = await extract_vector_from_url(img_url)
        if vector is not None:
            try:
                res = await features_coll.update_one( {"_id": category}, {"$addToSet": { "products": { "title": title, "image_url": img_url, "vector": vector.tolist() }}}, upsert=True )
                if res.upserted_id or res.modified_count > 0: added_count += 1; log.debug(f"DB changed for '{title}'")
                elif res.matched_count > 0: log.debug(f"DB: Product '{title}' likely exists.")
                else: log.warning(f"DB: update_one strange result for '{title}'.")
            except Exception as e: log.error(f"DB store error '{title}': {e}", exc_info=True)
    log.info(f"Append features for '{category}' from {site}: Processed {processed_count}, Added/Upserted {added_count}.")
    return added_count

# --- store_product_features using config ---
async def store_product_features(category: str, data: Dict, coreTitle: str, features_coll: AsyncIOMotorCollection):
    """Stores features for products (Original: Amz/Ebay only), uses passed collection."""
    if features_coll is None:
        log.error("DB collection missing for store_product_features.")
        return

    log.info(f"Storing features for cat:'{category}', title:'{coreTitle}' (Original Scope)...")
    total_added = 0
    amz = data.get("amazon", {}).get("products", [])
    eby = data.get("ebay", {}).get("products", [])
    ggl = data.get("google_shopping", {}).get("products", []) # Not stored in original logic
    ops = data.get("opensooq", {}).get("products", [])

    total_added += await appendToDoc(category, amz, coreTitle, features_coll)
    total_added += await appendToDoc(category, eby, coreTitle, features_coll)
    total_added += await appendToDoc(category, ggl, coreTitle, features_coll)
    total_added += await appendToDoc(category, ops, coreTitle, features_coll)

    log.info(f"Store complete for '{category}'. Total items Added/Upserted: {total_added}.")
# --- END OF FILE data_processing_final.py ---