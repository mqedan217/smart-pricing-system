# --- START OF FILE orchestrator.py ---

import asyncio
import json
import cv2
import logging
import sys
import os

# --- Determine Project Directory and Log File Path ---
try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILENAME = os.path.join(PROJECT_DIR, "project_activity.log")
except NameError:
     # __file__ might not be defined if run in certain interactive environments
     PROJECT_DIR = os.getcwd() # Fallback to current working directory
     LOG_FILENAME = os.path.join(PROJECT_DIR, "project_activity.log")
     print(f"Warning: Could not determine script directory, using CWD: {PROJECT_DIR}")


# --- Logging Setup (Configure Root Logger with Multiple Handlers) ---
# THIS ENTIRE BLOCK SHOULD BE PRESENT AND CORRECT
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
# --- End Logging Setup ---

# Get logger for this specific module AFTER root is configured
log = logging.getLogger(__name__)

# --- Graceful Imports ---
try:
    from image_recognition import detect_products, find_best_match, crop_selected_object, collection as features_collection
    from questionnaire import generate_next_question, identifyproduct
    # Ensure using the correct scraper filename you intend to use
    from scraper import scrape_all, store_input_features
    # Ensure using the correct data processing filename
    from data_processing import finalresults, store_product_features
    log.info("Successfully imported required modules.")
except ImportError as e:
    log.critical(f"Failed to import required modules: {e}. Cannot continue.", exc_info=True)
    sys.exit(1)
# ------------------------


# --- Main Orchestration Function (Keep the robust version) ---
async def run_orchestration():
    log.info("Starting main orchestration workflow...")

    # --- Step 1: Image Input and Reading ---
    image_path = r"C:\Users\Mogam\OneDrive\Desktop\images\samsung_galaxy_note_20_ultra_5g.jpg" # Ensure this path is correct
    image_bgr = None
    image_rgb = None
    try:
        log.info(f"Attempting to read image: {image_path}")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            log.critical(f"CRITICAL: Could not read image at {image_path}.")
            return # Exit gracefully
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        log.info("Image loaded and converted to RGB successfully.")
    except cv2.error as cv_err:
        log.critical(f"CRITICAL: OpenCV error processing image {image_path}: {cv_err}", exc_info=True)
        return
    except Exception as img_err:
        log.critical(f"CRITICAL: Unexpected error loading/converting image {image_path}: {img_err}", exc_info=True)
        return

    # --- Step 2: Object Detection and Cropping ---
    top_detection = None
    cropped_image = None
    detected_class = None # Initialize
    try:
        log.info("Detecting product in image...")
        top_detection = detect_products(image_path)

        if top_detection is None:
            log.error("No relevant product detected or detection failed."); return
        detected_class = top_detection.get("class")
        if not detected_class:
            log.error("Detection successful, but class name is missing."); return

        log.info(f"Detected: Class='{detected_class}', Confidence={top_detection.get('confidence', 0):.2f}")
        log.info("Cropping detected object...")
        cropped_image = crop_selected_object(image_rgb, top_detection)

        if cropped_image is None:
            log.error("Failed to crop the selected object."); return
        log.info("Object cropped successfully.")

    except Exception as detection_err:
         log.critical(f"CRITICAL: Error during product detection or cropping: {detection_err}", exc_info=True)
         return

    # --- Step 3: Feature Matching ---
    match_data = None; featuresVector = None; top_matches = None
    try:
        if features_collection is not None and cropped_image is not None:
            log.info(f"Finding potential matches in DB for category: {detected_class}...")
            match_data = find_best_match(cropped_image, detected_class, num_unique_results=3)
            if match_data:
                featuresVector = match_data.get("features_vector")
                top_matches = match_data.get("results")
                log.info(f"DB Match results: Found {len(top_matches) if top_matches else 0} matches.")
            else: log.warning("find_best_match returned None.")
        elif features_collection is None: log.warning("Skipping DB matching: No DB connection.")
        else: log.warning("Skipping DB matching: Cropping failed.")
    except Exception as matching_err:
         log.error(f"Error during feature matching: {matching_err}", exc_info=True)


    # --- Step 4 & 5: User Confirmation / Questionnaire ---
    confirmed_match_title = "None"; questionnaire_answers = []; ready_title = False
    try:
        if top_matches:
            print("\n--- Potential Matches ---")
            # ... (display matches logic kept same) ...
            for idx, match in enumerate(top_matches):
                 display_title = match.get('title', 'N/A')
                 similarity = match.get('similarity', 0.0)
                 print(f"{idx + 1}. {display_title[:75]}... (Sim: {similarity:.2f})")
            none_option_number = len(top_matches) + 1
            print(f"{none_option_number}. None of the above")

            print("-------------------------")
            while True: # User choice loop
                try:
                    choice_str = input(f"Select (1-{len(top_matches)}) or {none_option_number}: ").strip()
                    if not choice_str: continue
                    choice = int(choice_str)
                    if 1 <= choice <= len(top_matches):
                        confirmed_match_title = top_matches[choice - 1].get('title', "None") # user choice
                        ready_title = (confirmed_match_title != "None")
                        log.info(f"User confirmed match: '{confirmed_match_title}'")
                        break
                    elif choice == none_option_number: confirmed_match_title = "None"; log.info("User selected None."); break
                    else: print("Invalid choice.")
                except ValueError: print("Invalid input (number required).")
                except (KeyboardInterrupt, EOFError): log.warning("User interrupted confirmation."); confirmed_match_title = "None"; break
        else: log.info("No DB matches found/shown. Proceeding to questionnaire if needed.")

        if not ready_title:
            log.info("Starting questionnaire...")
            print("\n--- Product Questions ---")
            q_category = detected_class; q_counter = 0; MAX_QUESTIONS = 5 # Limit questions
            while q_counter < MAX_QUESTIONS:
                # ... (questionnaire logic kept same) ...
                 next_question_data = generate_next_question(q_category, questionnaire_answers)
                 if not next_question_data or "[Error]" in next_question_data.get("question", ""):
                     log.error(f"Failed to generate question {q_counter + 1}. Stopping questionnaire.")
                     print(f"\n[Error] Could not get question {q_counter + 1}. Proceeding...")
                     break
                 q_counter += 1; question = next_question_data.get("question", "N/A"); choices = next_question_data.get("choices", [])
                 print(f"\nQ{q_counter}: {question}");
                 if choices: print("Choices:", ", ".join(map(str, choices))[:100] + ("..." if len(", ".join(map(str, choices))) > 100 else ""))
                 while True: # Answer loop
                     try:
                         user_answer = input("Your answer: ").strip()
                         if user_answer: questionnaire_answers.append(user_answer); log.debug(f"Q{q_counter} Ans: {user_answer}"); break
                         else: print("Answer cannot be empty. Type 'skip' or provide details.") # Maybe allow skip?
                     except (KeyboardInterrupt, EOFError): log.warning("User interrupted questionnaire."); q_counter = MAX_QUESTIONS; break # Exit outer loop too
                     except Exception as e: log.error(f"Input error: {e}"); q_counter = MAX_QUESTIONS; break # Exit outer loop too
                 if q_counter >= MAX_QUESTIONS: break
            # log.info(f"Questionnaire answers: {questionnaire_answers}")
            print("-------------------------")

    except Exception as user_input_err:
        # log.error(f"Error during user input phase: {user_input_err}", exc_info=True)
        confirmed_match_title = "None"; ready_title = False # Reset state on error

    # --- Step 6: Identify Product Title ---
    specs = ""; coreTitle_insideQuotes = ""; fullTitle = ""
    try:
        specs = input("\nAdd any other specifications (e.g., color, condition) or press Enter: ").strip()
        print("specs are: ", specs)
        # log.info(f"User specs: '{specs}'")
        # log.info("Generating/Cleaning final product title...")
        title_input = confirmed_match_title if ready_title else questionnaire_answers
        print("questionnaire answers are: " )
        print(questionnaire_answers)
        print("title input is", title_input)
        title_result = identifyproduct(detected_class, title_input, specs, ready_title)
        print("title result is ", title_result)
        if not title_result or not title_result.get('product_title'):
            log.error("Failed to generate product title. Constructing fallback.")
            core_fallback = str(title_input) if isinstance(title_input, str) else " ".join(title_input)
            full_fallback = f'"{core_fallback}" {specs}'.strip()
            coreTitle_insideQuotes = core_fallback; fullTitle = full_fallback

        else:
            coreTitle_insideQuotes = title_result.get('core_title', '')
            fullTitle = title_result.get('product_title', '')
            if not coreTitle_insideQuotes and fullTitle and '"' in fullTitle and fullTitle.count('"') >= 2:
                 try: coreTitle_insideQuotes = fullTitle.split('"')[1]
                 except IndexError: coreTitle_insideQuotes = fullTitle

        if not fullTitle: log.critical("CRITICAL: Could not determine title for scraping."); return
        log.info(f"Final Core Title: '{coreTitle_insideQuotes}' | Full Title: '{fullTitle}'")
        print(f"\nOkay, searching for: {fullTitle}")

    except Exception as title_err:
        log.critical(f"CRITICAL: Error during title identification: {title_err}. Cannot proceed.", exc_info=True)
        return
    # # --- Step 7: Scrape Product Data ---
    scraped_data = None
    try:
        log.info("-----------------------------------------")
        log.info(f"--- Calling scrape_all with title: '{fullTitle}' ---")
        # Call scrape_all without explicit policy setting worries here
        scraped_data = await scrape_all(fullTitle)
        log.info("scrape_all call finished.")
        if scraped_data is None: # Check if scrape_all indicated failure
             log.error("Web scraping function returned None. Proceeding with empty data.")
             scraped_data = {"amazon_products": [], "ebay_products": [], "google_shopping": [], "opensooq_products": []}
        else:
            log.info(f" Scraped Counts - Amazon: {len(scraped_data.get('amazon_products',[]))}, eBay: {len(scraped_data.get('ebay_products',[]))}, Google: {len(scraped_data.get('google_shopping',[]))}, OpenSooq: {len(scraped_data.get('opensooq_products',[]))}")
    except NotImplementedError as nie: # Catch if it happens here
        log.critical(f"CRITICAL ASYNCIO ERROR during scrape_all call: {nie}", exc_info=True)
        log.critical("Default asyncio loop incompatible with Playwright subprocess.")
        scraped_data = {"amazon_products": [], "ebay_products": [], "google_shopping": [], "opensooq_products": []}
    except Exception as e:
        log.error(f"Error calling scrape_all from orchestrator: {e}", exc_info=True)
        scraped_data = {"amazon_products": [], "ebay_products": [], "google_shopping": [], "opensooq_products": []}
    log.info("-----------------------------------------")


    # --- Step 8: Process Prices ---
    processed_data = None
    try:
        log.info("Processing scraped data...")
        if not isinstance(scraped_data, dict):
             log.error("Scraped data is not a dictionary. Using empty data for processing.")
             scraped_data = {"amazon_products": [], "ebay_products": [], "google_shopping": [], "opensooq_products": []}
        # Assuming data_processing_final contains the correct finalresults
        processed_data = finalresults(scraped_data, fullTitle)
        if processed_data is None:
            log.error("finalresults returned None. Using default structure.")
            processed_data = { "amazon": {"products": [], "stats": {}}, "ebay": {"products": [], "stats": {}},
                               "google_shopping": {"products": [], "stats": {}}, "opensooq":{"products": [], "stats": []}, "overall_stats": {}, "related_products": [] }
        log.info("Price processing finished.")
    except Exception as process_err:
        log.error(f"Error during finalresults processing: {process_err}", exc_info=True)
        processed_data = { "amazon": {"products": [], "stats": {}}, "ebay": {"products": [], "stats": {}},
                           "google_shopping": {"products": [], "stats": {}}, "opensooq":{"products": [], "stats": []}, "overall_stats": {}, "related_products": [] }

    # --- Step 9: Display Results ---
    try:
        log.info("Displaying final results...")
        # Use original print statements
        amazon_products = processed_data.get("amazon", {}).get("products", [])
        ebay_products = processed_data.get("ebay", {}).get("products", [])
        google_shopping_products = processed_data.get("google_shopping", {}).get("products", [])
        opensooq_products = processed_data.get("opensooq", {}).get("products", [])
        amazon_stats = processed_data.get("amazon", {}).get("stats", {})
        ebay_stats = processed_data.get("ebay", {}).get("stats", {})
        google_shopping_stats = processed_data.get("google_shopping", {}).get("stats", {})
        opensooq_stats = processed_data.get("opensooq", {}).get("stats", {})
        overall_stats = processed_data.get("overall_stats", {})
        related_products = processed_data.get("related_products", [])

        print("\n--- Price Analysis Results ---") 
        print("amazon_products: ")
        print(json.dumps(amazon_products, indent=4, ensure_ascii=False))
        print("amazon_stats: ")
        print(json.dumps(amazon_stats, indent=4, ensure_ascii=False))
        print("ebay_products: ")
        print(json.dumps(ebay_products, indent=4, ensure_ascii=False))
        print("ebay_stats: ")
        print(json.dumps(ebay_stats, indent=4, ensure_ascii=False))
        print("google_products: ")
        print(json.dumps(google_shopping_products, indent=4, ensure_ascii=False))
        print("google_stats: ")
        print(json.dumps(google_shopping_stats, indent=4, ensure_ascii=False))
        print("openSooq_products: ")
        print(json.dumps(opensooq_products, indent=4, ensure_ascii=False))
        print("openSooq_stats: ")
        print(json.dumps(opensooq_stats, indent=4, ensure_ascii=False))
        print("overall stats: ")
        print(json.dumps(overall_stats, indent=4, ensure_ascii=False))
        print("related products: ")
        print(json.dumps(related_products, indent=4, ensure_ascii=False))

    except Exception as display_err:
        log.error(f"Error displaying results: {display_err}", exc_info=True)
        print("\nError encountered while displaying results. Check logs.")


    # --- Step 10: Store Features ---
    try:
        log.info("Storing features...") # Use log here for clarity
        if features_collection is not None:
            db_core_title = coreTitle_insideQuotes if coreTitle_insideQuotes else fullTitle
            if not db_core_title: log.warning("Cannot store features: No valid core title.")
            else:
                 if isinstance(processed_data, dict):
                      # Pass collection explicitly
                      store_product_features(detected_class, processed_data, db_core_title, features_coll=features_collection)
                 else: log.warning("Skipping scraped features storage: invalid data.")
                 if featuresVector is not None:
                     # Pass collection explicitly
                     store_input_features(featuresVector, db_core_title, detected_class, features_coll=features_collection)
                 else: log.info("Input image feature vector not available for storage.")
            log.info("Feature storage attempts finished.")
        else:
            log.warning("Skipping feature storage: MongoDB collection unavailable.")
    except Exception as db_store_err:
         log.error(f"Error during database storage phase: {db_store_err}", exc_info=True)
         print(f"Error during database storage: {db_store_err}") # Also print


    # --- Step 11: End Process ---
    log.info("Orchestration process completed.")
    print("\nProcessing complete.") # Use print as original


# --- Main Execution Block ---
if __name__ == "__main__":
    log.info("Starting Orchestrator Script (Policy Setting Removed)")

    # --- asyncio policy fix REMOVED ---
    # Relying on default event loop behavior for Python 3.12

    try:
        log.info("Running main orchestration async function...")
        asyncio.run(run_orchestration()) # Use the robust run_orchestration

    except KeyboardInterrupt:
        log.warning("Orchestration process interrupted by user (Ctrl+C).")
        print("\nProcess interrupted.")
    except NotImplementedError as nie:
         # Catch if default loop causes it during Playwright startup
         log.critical(f"CRITICAL ASYNCIO ERROR caught during run: {nie}", exc_info=True)
         print(f"\nCRITICAL ASYNCIO ERROR: {nie}. Default asyncio loop incompatible with Playwright subprocess.")
    except Exception as main_err:
         log.critical(f"An unhandled error occurred in the main orchestrator block: {main_err}", exc_info=True)
         print(f"\nAn unhandled critical error occurred: {main_err}. Check logs for details.")
    finally:
         log.info("Orchestrator script finished execution block.")
# --- END OF FILE orchestrator.py ---