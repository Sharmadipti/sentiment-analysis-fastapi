# Sentiment Analysis API using NLP and FastAPI

## Project Overview
An end-to-end sentiment analysis system that predicts whether a given piece
of text is positive or negative. The model is trained with classical NLP
techniques (TF-IDF + Logistic Regression) and served as a REST API using
FastAPI.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- FastAPI
- Uvicorn
- Docker

---

## Project Structure
```
.
├── app.py                  # FastAPI app / inference endpoint
├── train.py                # Trains the model and saves artifacts
├── preprocessing.py         # Text cleaning shared by training + inference
├── data/
│   └── reviews.csv         # Training data (text, label)
├── sentiment_model.pkl      # Trained Logistic Regression model
├── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── requirements.txt
└── Dockerfile
```

---

## Machine Learning Workflow
1. Text cleaning (lowercase, strip URLs/HTML/punctuation) — `preprocessing.py`
2. Text vectorization using TF-IDF (unigrams + bigrams)
3. Model training using Logistic Regression
4. Evaluation using accuracy, precision, recall, and F1-score
5. Saving the trained model and vectorizer with `joblib`
6. Serving the model as a REST API using FastAPI, with the **same**
   cleaning function applied at inference time as at training time

### About the dataset
`data/reviews.csv` contains **25,000 real IMDB movie reviews** (12,500
positive / 12,500 negative, balanced), sourced from the Large Movie
Review Dataset (Maas et al.). Columns: `text`, `label`.

On a held-out 20% test split, the TF-IDF + Logistic Regression pipeline
in `train.py` achieves:

```
Accuracy: 0.885
              precision    recall  f1-score
negative         0.89       0.87      0.88
positive         0.88       0.90      0.89
```

`train.py` will work unchanged with any other dataset that has the same
two columns (`text`, `label`), if you want to try a different domain
(e.g. product or app reviews instead of movies).

`generate_dataset.py` is also included — it was used during early
development to create a small synthetic dataset for quickly testing the
pipeline end-to-end before plugging in real data. It's not used for the
current model but kept for reference.

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the model — regenerates the .pkl files
python train.py

# 3. Start the API
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Running with Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## API Endpoints

### GET `/`
Health check — confirms the API is running.
```json
{ "message": "Sentiment Analysis API is running" }
```

### GET `/health`
Confirms the model and vectorizer are actually loaded in memory.
```json
{ "status": "ok" }
```

### POST `/predict`
Predicts the sentiment of a piece of text.

**Request:**
```json
{ "text": "This product exceeded all my expectations!" }
```

**Response:**
```json
{ "sentiment": "positive", "input_text": "This product exceeded all my expectations!" }
```

Returns `400` for empty/whitespace-only input, and `503` if the model
failed to load on startup.

---

## Deployment
This app is deployable as-is to any platform that supports Docker or a
Python web service, e.g. Render, Railway, or Hugging Face Spaces (Docker
SDK). Set the start command to:
```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## Possible Improvements
- Add automated tests (`pytest` + `httpx`/`TestClient`)
- Add rate limiting / auth if exposed publicly
- Track experiments (accuracy per model version) as the dataset changes
- Swap classical TF-IDF pipeline for a transformer-based model (e.g.
  DistilBERT) for higher accuracy on nuanced text
