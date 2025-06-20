# --- START OF FILE image_recognition.py ---

import os
import logging
import cv2
import numpy as np
import faiss
import torch
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50 # Keep specific model import
# Import specific weights enum type if needed for type hinting or direct access
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
from pymongo import MongoClient, errors as mongo_errors
from typing import Optional, Dict, Any, List, Tuple # For type hints
from motor.motor_asyncio import AsyncIOMotorCollection
# --- Import Configuration ---
import config # Import the config module
# ---------------------------

# --- Environment Variable Fix (Keep this at the top) ---
# Should KMP_DUPLICATE_LIB_OK also be configurable? Probably not needed.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ---------------------------------------------------------
# --- Logging Setup ---
# Get logger for this module. Assumes root logger is configured elsewhere (e.g., orchestrator)
log = logging.getLogger(__name__)
# ---------------------
# --- Load YOLO ---
try:
    yolo_model_path_str = str(config.YOLO_MODEL_PATH) # Get path from config
    if config.YOLO_MODEL_PATH.is_file():
        log.info(f"Loading YOLO model from: {yolo_model_path_str}")
        YOLO_MODEL = YOLO(yolo_model_path_str)
        log.info("✅ YOLO model loaded successfully.")
    else:
        log.error(f"❌ CRITICAL: YOLO model file not found at: {yolo_model_path_str}")
except AttributeError:
    log.error("❌ CRITICAL: YOLO_MODEL_PATH not defined in config.")
except Exception as e:
    log.error(f"❌ CRITICAL: Failed to load YOLO model: {e}", exc_info=True)

# ---------------------
# --- Load ResNet50 using config ---
RESNET_LOADED: bool = False
try:
    # Determine the correct weights enum from the config string
    weights_enum_name = config.RESNET_WEIGHTS_STR.upper()
    weights_enum = getattr(ResNet50_Weights, weights_enum_name, ResNet50_Weights.DEFAULT)
    # Validate that we got the correct type
    if not isinstance(weights_enum, ResNet50_Weights):
        log.warning(f"Configured ResNet weights '{config.RESNET_WEIGHTS_STR}' invalid. Using DEFAULT.")
        weights_enum = ResNet50_Weights.DEFAULT

    log.info(f"Loading ResNet50 model with weights: {weights_enum.name}")
    temp_model = resnet50(weights=weights_enum) # Load with specified weights
    temp_model.eval()
    RESNET_MODEL = torch.nn.Sequential(*list(temp_model.children())[:-1])  # remove final FC

    # Standard transform, usually not configured externally unless necessary
    RESNET_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    RESNET_LOADED = True
    log.info("ResNet50 model and transform loaded successfully.")
except AttributeError:
     log.error(f"Invalid ResNet weights name specified in config: '{config.RESNET_WEIGHTS_STR}'. Using DEFAULT.", exc_info=True)
     # Attempt load with default if specific name fails
     try:
         RESNET_MODEL = resnet50(weights=ResNet50_Weights.DEFAULT)
         RESNET_MODEL.eval()
         RESNET_MODEL = torch.nn.Sequential(*list(RESNET_MODEL.children())[:-1])
         RESNET_TRANSFORM = transforms.Compose([...]) # Re-add transform definition
         RESNET_LOADED = True
         log.info("ResNet50 loaded with DEFAULT weights after config error.")
     except Exception as e_fallback:
          log.error(f"Failed to load ResNet50 model even with DEFAULT weights: {e_fallback}", exc_info=True)
except Exception as e:
    log.error(f"Failed to load ResNet50 model: {e}", exc_info=True)
# -------------------------------------------------

# --- Detect Products Function using config ---
def detect_products(image_array: np.ndarray) -> Optional[Dict[str, Any]]:
    """Detects objects using YOLO model path from config, taking numpy array"""

    if YOLO_MODEL is None:
        log.error("YOLO model not loaded. Cannot perform detection.")
        return None
    
    
    if image_array is None or image_array.size ==0:
        log.error("Input image array is empty or None for detection.")
        return None
    detections = []
    try:
        results = YOLO_MODEL(image_array)
        log.info("YOLO detection performed.")
    except Exception as e:
        log.error(f"Error during YOLO detection on image array : {e}", exc_info=True)
        return None

    # Process results
    for result in results:
        try:
            if not hasattr(result, 'boxes') or result.boxes is None: continue
            for box in result.boxes:
                try:
                    xyxy = box.xyxy[0]; conf = box.conf[0]; cls = box.cls[0]
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    conf_val = conf.item(); cls_id = int(cls.item())
                    class_name = result.names.get(cls_id) if hasattr(result,'names') and isinstance(result.names,dict) else None
                    if class_name is None: continue # skip if class name cannot be determined

                    # Use irrelevant classes set from config
                    if class_name.lower() not in config.IRRELEVANT_CLASSES:
                        detections.append({
                            "class": class_name,
                            "confidence": float(conf_val),
                            "box": (x1, y1, x2, y2)
                        })
                except Exception as box_err:
                    log.warning(f"Error processing a detection box: {box_err}", exc_info=False) # Less verbose
                    continue
        except Exception as res_err:
            log.warning(f"Error processing a result object: {res_err}", exc_info=False)
            continue

    if not detections:
        log.warning("No relevant objects detected after filtering.")
        return None

    # Sort and return top detection
    try:
        # Sort by confidence DESCENDING
        detections.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        top_detection = detections[0]
        log.info(f"Top detection: {top_detection.get('class','N/A')} (Conf: {top_detection.get('confidence', 0.0):.2f})")
        return top_detection
    except (IndexError, KeyError, Exception) as sort_err:
         log.error(f"Error sorting or accessing top detection: {sort_err}", exc_info=True)
         return None


# --- Crop Object Function using config ---
def crop_selected_object(image_array: np.ndarray, selected_detection: Dict[str, Any]) -> Optional[np.ndarray]:
    """Crops image using coordinates from selected detection."""
    # ... (Initial checks for image_array, selected_detection remain same) ...
    if image_array is None: log.error("Crop input image is None."); return None
    if selected_detection is None or "box" not in selected_detection: log.error("Crop detection invalid."); return None

    try:
        x1, y1, x2, y2 = selected_detection["box"]
        h, w = image_array.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x1 >= x2 or y1 >= y2: log.warning(f"Invalid box dims after clip: {x1},{y1} -> {x2},{y2}"); return None

        cropped_image = image_array[y1:y2, x1:x2]
        if cropped_image.size == 0: log.warning("Cropped image is empty."); return None

        return cropped_image

    except Exception as crop_err:
         log.error(f"Error during image cropping: {crop_err}", exc_info=True)
         return None

# --- Feature Extraction (no changes needed, uses global model/transform) ---
def extract_features(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Extract features using globally loaded model and transform."""
    if not RESNET_LOADED:
        log.error("Cannot extract features: ResNet components not loaded.")
        return None
    if image_np is None or image_np.size == 0:
        log.error("Cannot extract features from empty image.")
        return None
    try:
        image = Image.fromarray(image_np)
        tensor = RESNET_TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            vec = RESNET_MODEL(tensor).squeeze().numpy()
        # log.debug("Feature extraction successful.") # Reduce noise maybe
        return vec.astype("float32")
    except Exception as e:
        log.error(f"Error during feature extraction: {e}", exc_info=True)
        return None

# --- Find Best Match using config ---
# Use config values as defaults for parameters
async def find_best_match(
        features_coll: AsyncIOMotorCollection,
        cropped_image: np.ndarray, category: str,
        num_unique_results: int = config.IMG_REC_NUM_MATCHES,
        search_k_multiplier: int = config.IMG_REC_K_MULTIPLIER,
                     
                     ) -> Dict[str, Any]:
    """Finds matches using DB collection ASYNCHRONOUSLY."""

    # Default return structure
    default_return = {"results": None, "features_vector": None}

    if not RESNET_LOADED:
        log.error("Cannot find match: ResNet model not loaded.")
        return default_return
    # Use the collection object initialized using config values
    if features_coll is None:
         log.error("Cannot find match: MongoDB collection unavailable.")
         return default_return

    # Extract feature vector
    query_vector = extract_features(cropped_image)
    if query_vector is None:
        log.error("Feature extraction failed for the query image. Cannot find match.")
        # Still return default, but without features_vector
        return default_return
    # Update default return to include the vector now that we have it
    default_return["features_vector"] = query_vector.tolist() # Convert to list for JSON potentially

    query_vector_np = query_vector.reshape(1, -1)

    # Get document from DB
    document = None
    try:
        document = await features_coll.find_one({"_id": category})
    except mongo_errors.PyMongoError as db_err:
         log.error(f"MongoDB error finding category '{category}': {db_err}", exc_info=True)
         return default_return # Return default (with vector)

    if not document or "products" not in document or not document["products"]:
        log.warning(f"No products found in DB for category: {category}")
        return default_return

    # Prepare vectors (keep robust filtering)
    product_vectors = []
    product_info = []
    try:
        valid_products = [prod for prod in document.get("products", []) if isinstance(prod.get("vector"), list) and len(prod["vector"]) > 0]
        if not valid_products:
            log.warning(f"No valid vectors in category: {category}"); return default_return
        product_vectors = np.array([prod["vector"] for prod in valid_products]).astype('float32')
        product_info = valid_products
        log.info(f"Loaded {len(product_info)} vectors for category '{category}'.")
    except Exception as e:
         log.error(f"Error preparing DB vectors for '{category}': {e}", exc_info=True)
         return default_return

    # FAISS Search (keep robust logic)
    try:
        dimension = product_vectors.shape[1]
        if dimension != query_vector_np.shape[1]:
             log.error(f"Dimension mismatch: DB {dimension} vs Query {query_vector_np.shape[1]}"); return default_return
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(product_vectors)

        # Use parameters (potentially from config via function defaults)
        k_to_search = min(max(1, num_unique_results * search_k_multiplier), faiss_index.ntotal)
        log.info(f"FAISS search k={k_to_search} (num_results={num_unique_results}, multiplier={search_k_multiplier})")
        distances, indices = faiss_index.search(query_vector_np, k_to_search)

        if indices.size == 0: log.warning("FAISS returned no results."); return default_return
        log.info(f"FAISS found {indices.shape[1]} potential matches.")
    except Exception as e:
        log.error(f"FAISS error: {e}", exc_info=True)
        return default_return

    # --- Process and Filter for Uniqueness ---
    unique_results = []
    seen_titles = set()
    actual_l2_distances = []
    if indices.size > 0 and distances.size > 0:
        actual_l2_distances = distances[0].tolist() # Convert to simple list of floats

    d_min = min(actual_l2_distances) if actual_l2_distances else 0.0
    d_max = max(actual_l2_distances) if actual_l2_distances else 0.0
    distance_range = d_max - d_min
    try:
        # Iterate using the found indices and distances
        for i, dist in zip(indices[0], distances[0]):
            # Stop if we have enough unique results
            if len(unique_results) >= num_unique_results:
                break

            # Get corresponding product info using the index 'i'
            if not (0 <= i < len(product_info)): # Check bounds
                log.warning(f"FAISS returned invalid index {i} (out of bounds for {len(product_info)} products). Skipping.")
                continue # Skip invalid index

            matched_product = product_info[i]

            # Check if essential keys exist
            title = matched_product.get("title")
            image_url = matched_product.get("image_url")

            if not title or not image_url:
                log.warning(f"Product entry at index {i} is missing 'title' or 'image_url'. Skipping. Data: {matched_product}")
                continue

            # Add if the title hasn't been seen before
            if title not in seen_titles:
                similarity = 0.0 # Default
                try:
                    if not actual_l2_distances: # Should not happen if we got here
                        similarity = 0.0
                    elif distance_range < 1e-6:  # All distances are effectively the same
                        similarity = 1.0 if abs(dist - d_min) < 1e-6 else 0.0 # 1.0 if it's the min, else 0
                        if len(actual_l2_distances) == 1: similarity = 1.0 # If only one result, it's the best
                    else:
                        # Normalized distance (0 for best/d_min, 1 for worst/d_max)
                        normalized_dist_score = (dist - d_min) / distance_range
                        similarity = 1.0 - normalized_dist_score # Invert to get similarity

                    similarity = max(0.0, min(1.0, similarity)) # Clamp

                except Exception as sim_err: # Catch any math errors
                     log.warning(f"Could not calculate min-max similarity for distance {dist}: {sim_err}. Product {title}. Assigning 0.")
                     similarity = 0.0

                unique_results.append({
                    "title": title,
                    "image_url": image_url,
                    "similarity": round(similarity,2)
                })
                seen_titles.add(title)
                log.debug(f"Added unique match: {title} (Similarity: {similarity:.4f}, Dist: {dist:.4f})")

    except Exception as proc_err:
         log.error(f"Error processing FAISS results: {proc_err}", exc_info=True)
         # Return whatever unique results were found so far, or None if none
         return {"results": unique_results if unique_results else None, "features_vector": query_vector}

    log.info(f"Found {len(unique_results)} unique matches.")
    return {"results": unique_results if unique_results else None, "features_vector": query_vector.tolist()}
# --- END OF FILE image_recognition.py ---


# version for ochestrator below

# --- START OF FILE image_recognition_final.py ---

# import os
# import logging
# import cv2
# import numpy as np
# import faiss
# import torch
# from ultralytics import YOLO
# from torchvision import transforms
# from torchvision.models import resnet50 # Keep specific model import
# # Import specific weights enum type if needed for type hinting or direct access
# from torchvision.models.resnet import ResNet50_Weights
# from PIL import Image
# from pymongo import MongoClient, errors as mongo_errors
# from typing import Optional, Dict, Any, List, Tuple # For type hints

# # --- Import Configuration ---
# import config # Import the config module
# # ---------------------------

# # --- Environment Variable Fix (Keep this at the top) ---
# # Should KMP_DUPLICATE_LIB_OK also be configurable? Probably not needed.
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # ---------------------------------------------------------

# # --- Logging Setup ---
# # Get logger for this module. Assumes root logger is configured elsewhere (e.g., orchestrator)
# log = logging.getLogger(__name__)
# # ---------------------

# # --- Constants from Config (Replaces hardcoded definitions) ---
# # IRRELEVANT_CLASSES is already loaded as config.IRRELEVANT_CLASSES (Set[str])
# # -------------------------------------------------------------

# # --- MongoDB Client Initialization using config ---
# client: Optional[MongoClient] = None
# db = None
# collection = None
# features_collection = None # Use this name to match orchestrator import alias

# # Check if FEATURES_DB_URI was successfully loaded by config.py
# if config.DEFAULT_MONGO_URI:
#     try:
#         client = MongoClient(config.DEFAULT_MONGO_URI, serverSelectionTimeoutMS=5000, appName="ImageRecognition")
#         client.admin.command('ping') # Verify connection
#         # Use names from config
#         db = client[config.FEATURES_DB_NAME]
#         collection = db[config.FEATURES_COLLECTION_NAME]
#         features_collection = collection # Assign to the name orchestrator expects
#         log.info(f"MongoDB connected via config: {config.FEATURES_DB_NAME}/{config.FEATURES_COLLECTION_NAME}")
#     except mongo_errors.ConnectionFailure as e:
#         log.error(f"MongoDB connection failed using URI from config: {config.FEATURES_DB_URI}", exc_info=True)
#         # client, db, collection, features_collection remain None
#     except Exception as e:
#         log.error(f"Unexpected error connecting to MongoDB via config", exc_info=True)
#         # client, db, collection, features_collection remain None
# else:
#     log.error("MongoDB URI not found in configuration. Database features disabled.")
# # -----------------------------------------------

# # --- Load ResNet50 using config ---
# model: Optional[torch.nn.Module] = None
# transform: Optional[transforms.Compose] = None
# RESNET_LOADED: bool = False
# try:
#     # Determine the correct weights enum from the config string
#     weights_enum_name = config.RESNET_WEIGHTS_STR.upper()
#     weights_enum = getattr(ResNet50_Weights, weights_enum_name, ResNet50_Weights.DEFAULT)
#     # Validate that we got the correct type
#     if not isinstance(weights_enum, ResNet50_Weights):
#         log.warning(f"Configured ResNet weights '{config.RESNET_WEIGHTS_STR}' invalid. Using DEFAULT.")
#         weights_enum = ResNet50_Weights.DEFAULT

#     log.info(f"Loading ResNet50 model with weights: {weights_enum.name}")
#     model = resnet50(weights=weights_enum) # Load with specified weights
#     model.eval()
#     model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC

#     # Standard transform, usually not configured externally unless necessary
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     RESNET_LOADED = True
#     log.info("ResNet50 model and transform loaded successfully.")
# except AttributeError:
#      log.error(f"Invalid ResNet weights name specified in config: '{config.RESNET_WEIGHTS_STR}'. Using DEFAULT.", exc_info=True)
#      # Attempt load with default if specific name fails
#      try:
#          model = resnet50(weights=ResNet50_Weights.DEFAULT)
#          model.eval(); model = torch.nn.Sequential(*list(model.children())[:-1])
#          transform = transforms.Compose([...]) # Re-add transform definition
#          RESNET_LOADED = True
#          log.info("ResNet50 loaded with DEFAULT weights after config error.")
#      except Exception as e_fallback:
#           log.error(f"Failed to load ResNet50 model even with DEFAULT weights: {e_fallback}", exc_info=True)
# except Exception as e:
#     log.error(f"Failed to load ResNet50 model: {e}", exc_info=True)
# # -------------------------------------------------

# # --- Detect Products Function using config ---
# def detect_products(image_path: str) -> Optional[Dict[str, Any]]:
#     """Detects objects using YOLO model path from config."""
#     detections = []
#     model_yolo = None
#     try:
#         # Use YOLO_MODEL_PATH from config (it's a Path object)
#         yolo_model_file = str(config.YOLO_MODEL_PATH)
#         # Check if file exists before loading
#         if not config.YOLO_MODEL_PATH.is_file():
#              log.error(f"YOLO model file not found at configured path: {yolo_model_file}")
#              return None
#         model_yolo = YOLO(yolo_model_file)
#         log.info(f"YOLOv8 model loaded from config path: {yolo_model_file}")
#     except Exception as e:
#         log.error(f"Failed to load YOLOv8 model from {config.YOLO_MODEL_PATH}: {e}", exc_info=True)
#         return None

#     try:
#         results = model_yolo(image_path)
#         log.info(f"YOLO detection performed on: {image_path}")
#     except Exception as e:
#         log.error(f"Error during YOLO detection on {image_path}: {e}", exc_info=True)
#         return None

#     # Process results
#     for result in results:
#         try:
#             if not hasattr(result, 'boxes') or result.boxes is None: continue
#             for box in result.boxes:
#                 try:
#                     xyxy = box.xyxy[0]; conf = box.conf[0]; cls = box.cls[0]
#                     x1, y1, x2, y2 = map(int, xyxy.tolist())
#                     conf_val = conf.item(); cls_id = int(cls.item())

#                     if hasattr(result, 'names') and isinstance(result.names, dict) and cls_id in result.names:
#                          class_name = result.names[cls_id]
#                     else: continue

#                     # Use irrelevant classes set from config
#                     if class_name.lower() not in config.IRRELEVANT_CLASSES:
#                         detections.append({
#                             "class": class_name,
#                             "confidence": float(conf_val),
#                             "box": (x1, y1, x2, y2)
#                         })
#                 except Exception as box_err:
#                     log.warning(f"Error processing a detection box: {box_err}", exc_info=False) # Less verbose
#                     continue
#         except Exception as res_err:
#             log.warning(f"Error processing a result object: {res_err}", exc_info=False)
#             continue

#     if not detections:
#         log.warning("No relevant objects detected after filtering.")
#         return None

#     # Sort and return top detection
#     try:
#         # Sort by confidence DESCENDING
#         detections.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
#         top_detection = detections[0]
#         log.info(f"Top detection: {top_detection.get('class','N/A')} (Conf: {top_detection.get('confidence', 0.0):.2f})")
#         return top_detection
#     except (IndexError, KeyError, Exception) as sort_err:
#          log.error(f"Error sorting or accessing top detection: {sort_err}", exc_info=True)
#          return None


# # --- Crop Object Function using config ---
# def crop_selected_object(image_array: np.ndarray, selected_detection: Dict[str, Any]) -> Optional[np.ndarray]:
#     """Crops image using coordinates from selected detection."""
#     # ... (Initial checks for image_array, selected_detection remain same) ...
#     if image_array is None: log.error("Crop input image is None."); return None
#     if selected_detection is None or "box" not in selected_detection: log.error("Crop detection invalid."); return None

#     try:
#         x1, y1, x2, y2 = selected_detection["box"]
#         h, w = image_array.shape[:2]
#         x1, y1 = max(0, int(x1)), max(0, int(y1))
#         x2, y2 = min(w, int(x2)), min(h, int(y2))
#         if x1 >= x2 or y1 >= y2: log.warning(f"Invalid box dims after clip: {x1},{y1} -> {x2},{y2}"); return None

#         cropped_image = image_array[y1:y2, x1:x2]
#         if cropped_image.size == 0: log.warning("Cropped image is empty."); return None

#         return cropped_image

#     except Exception as crop_err:
#          log.error(f"Error during image cropping: {crop_err}", exc_info=True)
#          return None

# # --- Feature Extraction (no changes needed, uses global model/transform) ---
# def extract_features(image_np: np.ndarray) -> Optional[np.ndarray]:
#     """Extract features using globally loaded model and transform."""
#     if not RESNET_LOADED or model is None or transform is None:
#         log.error("Cannot extract features: ResNet components not loaded.")
#         return None
#     if image_np is None or image_np.size == 0:
#         log.error("Cannot extract features from empty image.")
#         return None
#     try:
#         image = Image.fromarray(image_np)
#         tensor = transform(image).unsqueeze(0)
#         with torch.no_grad():
#             vec = model(tensor).squeeze().numpy()
#         # log.debug("Feature extraction successful.") # Reduce noise maybe
#         return vec.astype("float32")
#     except Exception as e:
#         log.error(f"Error during feature extraction: {e}", exc_info=True)
#         return None

# # --- Find Best Match using config ---
# # Use config values as defaults for parameters
# def find_best_match(cropped_image: np.ndarray, category: str,
#                      num_unique_results: int = config.IMG_REC_NUM_MATCHES,
#                      search_k_multiplier: int = config.IMG_REC_K_MULTIPLIER
#                      ) -> Dict[str, Any]:
#     """Finds matches using DB collection from config and parameters from config."""

#     # Default return structure
#     default_return = {"results": None, "features_vector": None}

#     if not RESNET_LOADED:
#         log.error("Cannot find match: ResNet model not loaded.")
#         return default_return
#     # Use the collection object initialized using config values
#     if collection is None:
#          log.error("Cannot find match: MongoDB collection unavailable (check config and connection).")
#          return default_return

#     # Extract feature vector
#     query_vector = extract_features(cropped_image)
#     if query_vector is None:
#         log.error("Feature extraction failed for the query image. Cannot find match.")
#         # Still return default, but without features_vector
#         return default_return
#     # Update default return to include the vector now that we have it
#     default_return["features_vector"] = query_vector.tolist() # Convert to list for JSON potentially

#     query_vector_np = query_vector.reshape(1, -1)

#     # Get document from DB
#     document = None
#     try:
#         document = collection.find_one({"_id": category})
#     except mongo_errors.PyMongoError as db_err:
#          log.error(f"MongoDB error finding category '{category}': {db_err}", exc_info=True)
#          return default_return # Return default (with vector)

#     if not document or "products" not in document or not document["products"]:
#         log.warning(f"No products found in DB for category: {category}")
#         return default_return

#     # Prepare vectors (keep robust filtering)
#     product_vectors = []
#     product_info = []
#     try:
#         valid_products = [prod for prod in document.get("products", []) if isinstance(prod.get("vector"), list) and len(prod["vector"]) > 0]
#         if not valid_products:
#             log.warning(f"No valid vectors in category: {category}"); return default_return
#         product_vectors = np.array([prod["vector"] for prod in valid_products]).astype('float32')
#         product_info = valid_products
#         log.info(f"Loaded {len(product_info)} vectors for category '{category}'.")
#     except Exception as e:
#          log.error(f"Error preparing DB vectors for '{category}': {e}", exc_info=True)
#          return default_return

#     # FAISS Search (keep robust logic)
#     try:
#         dimension = product_vectors.shape[1]
#         if dimension != query_vector_np.shape[1]:
#              log.error(f"Dimension mismatch: DB {dimension} vs Query {query_vector_np.shape[1]}"); return default_return
#         faiss_index = faiss.IndexFlatL2(dimension)
#         faiss_index.add(product_vectors)

#         # Use parameters (potentially from config via function defaults)
#         k_to_search = min(max(1, num_unique_results * search_k_multiplier), faiss_index.ntotal)
#         log.info(f"FAISS search k={k_to_search} (num_results={num_unique_results}, multiplier={search_k_multiplier})")
#         distances, indices = faiss_index.search(query_vector_np, k_to_search)

#         if indices.size == 0: log.warning("FAISS returned no results."); return default_return
#         log.info(f"FAISS found {indices.shape[1]} potential matches.")
#     except Exception as e:
#         log.error(f"FAISS error: {e}", exc_info=True)
#         return default_return

#     # Process results (keep robust logic)
#     unique_results = []
#     seen_titles = set()
#     try:
#         for i, dist in zip(indices[0], distances[0]):
#             if len(unique_results) >= num_unique_results: break # Use parameter
#             if not (0 <= i < len(product_info)): continue
#             matched_product = product_info[i]
#             title = matched_product.get("title"); image_url = matched_product.get("image_url")
#             if not title or not image_url: continue

#             if title not in seen_titles:
#                 try:
#                     similarity = max(0.0, min(1.0, 1.0 / (1.0 + float(dist))))
#                     unique_results.append({"title": title, "image_url": image_url, "similarity": similarity})
#                     seen_titles.add(title)
#                 except (ValueError, TypeError): continue # Skip if dist invalid
#     except Exception as proc_err:
#          log.error(f"Error processing FAISS results: {proc_err}", exc_info=True)
#          # Return partial results if any were processed
#          return {"results": unique_results if unique_results else None, "features_vector": query_vector.tolist()}

#     log.info(f"Found {len(unique_results)} unique matches.")
#     # Return final results
#     return {"results": unique_results if unique_results else None, "features_vector": query_vector.tolist()}

# # # --- END OF FILE image_recognition_final.py ---