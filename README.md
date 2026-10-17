# Etsawakli: Smart Object Detection and Price Prediction for E-Commerce

**Graduation project, B.Sc. Computer Engineering, University of Jordan (2025). Grade: A.**
Supervised by Prof. Ramzi Saifan.

Take a photo of a product. Etsawakli identifies it, collects live prices from Amazon, eBay, Google Shopping and OpenSooq, filters out irrelevant and anomalous listings, forecasts where the price is heading, and alerts you when it drops.

<p align="center">
  <img src="docs/images/app_upload.jpg" width="32%">
  <img src="docs/images/app_questionnaire.jpg" width="32%">
  <img src="docs/images/app_notification.jpg" width="32%">
</p>

## The problem

Comparing prices by hand across many sites is slow, dynamic pricing makes it hard to know when to buy, and identifying a product from a photo or a vague idea is hard. Etsawakli handles all of this in one flow: snap a photo, confirm the product, get the market picture.

## How it works

<p align="center"><img src="docs/images/system_workflow.jpg" width="45%"></p>

### 1. Product recognition
- **Detection**: YOLOv8m pretrained on Open Images V7 (~600 classes) locates the main object and filters out irrelevant classes such as people and animals.
- **Visual matching**: ResNet-50 embeds the cropped object, and a FAISS index returns the three most similar unique products from the database.
- **Questionnaire fallback**: when there's no match, or the user rejects the matches, GPT-4o-mini asks up to 5 context-aware questions, from general to specific. It then merges the answers into a clean "core title" optimised for search. Question paths are cached in MongoDB, which cuts LLM calls and cost on common paths.

![YOLOv8 detections](docs/images/detections.jpg)

### 2. Data retrieval
A hybrid strategy, chosen per site:
- **Playwright** for Amazon, eBay and OpenSooq. These sites rely heavily on JavaScript, and the CSS selectors are stored in the database so they can be updated without a redeploy.
- **BrightData SERP API** for Google Shopping, whose anti-bot measures make direct scraping unreliable.

### 3. Cleaning and analysis
![Processing pipeline](docs/images/processing_pipeline.jpg)

Prices are normalised to USD. Listings are scored for relevance using Word2Vec title embeddings with FAISS, and the top matches are expanded and passed through an IQR filter that removes price outliers.

![Anomaly detection on iPhone 13 listings](docs/images/price_anomalies.jpg)
*Amazon results for an iPhone 13 search: red diamonds are listings the pipeline flagged as irrelevant (other models, accessories) before computing price statistics.*

### 4. Price forecasting
TimeGPT (Nixtla) forecasts future prices zero-shot from historical prices aggregated monthly. Monthly aggregation smooths out daily noise while still giving actionable signals.

![Price forecast](docs/images/price_forecast.jpg)

### 5. Price-drop notifier
APScheduler rechecks tracked items every 24 hours, and Firebase Cloud Messaging sends a push notification when a price falls by 5% or more. The threshold is high enough to matter and low enough to catch real deals. The interval balances timely updates against scraping load and rate limits.

## Performance

Average response time per endpoint:

| Endpoint | Avg (ms) |
|---|---|
| `GET /scrape-status/{session_id}` | 10.88 |
| `POST /predict-prices/{session_id}` | 16.13 |
| `POST /scrape/{session_id}` | 16.70 |
| `POST /login/` | 484.68 |
| `POST /signup/` | 528.05 |
| `GET /predict-prices/status/{session_id}` | 546.38 |
| `POST /detect/` | 555.00 |
| `POST /answer/` | 1022.88 |
| **Aggregated** | **627.51** |

Scraping and forecasting run as background tasks, so those endpoints return immediately while the client polls for status.

## The app

<p align="center">
  <img src="docs/images/app_matches.jpg" width="40%">
  <img src="docs/images/app_auth.jpg" width="55%">
</p>

![Results screens](docs/images/app_results.jpg)

The Flutter app (Android first) lets users sign up and log in, upload and crop a product photo, pick a match or answer the questionnaire, then browse results by marketplace with overall statistics and a price history chart.

## Stack

**Backend:** FastAPI, MongoDB (Motor), APScheduler, Firebase Admin
**ML / AI:** Ultralytics YOLOv8, PyTorch / torchvision (ResNet-50), FAISS, Gensim Word2Vec, OpenAI API, Nixtla TimeGPT
**Data collection:** Playwright, BrightData SERP API
**Frontend:** Flutter (not included in this repo)

## Team

| | |
|---|---|
| **Marwan Qedan** | Backend and AI pipeline: object detection, visual matching, LLM questionnaire, scraping, data processing and anomaly filtering, notifier, FastAPI service |
| **Albatoul Alfassi** | Flutter mobile app and the TimeGPT price predictor |

## Running it

```bash
cd backend
pip install -r requirements.txt
playwright install
```

1. Copy `global_config.example.json`, fill in your own API keys, and insert it into MongoDB as document `global_config` in `app_config.settings`. The backend reads its config from there, via `CONFIG_MONGO_URI`, which defaults to localhost.
2. Run `uvicorn main:app --reload`.

YOLO weights download automatically on first run.

## Limitations

- Scraper selectors break when sites change their layout.
- The system depends on third-party AI APIs for cost, uptime and output consistency.
- Data quality varies a lot across marketplaces.

Future work: more marketplaces, user feedback loops, personalised recommendations, and LLM-driven semantic scraping in place of fixed selectors.
