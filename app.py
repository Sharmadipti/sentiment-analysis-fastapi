import logging

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocessing import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api")

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

app = FastAPI(
    title="Sentiment Analysis API",
    description="TF-IDF + Logistic Regression sentiment classifier.",
    version="1.0.0",
)

# Allow the API to be called from a browser-based frontend.
# Replace "*" with your actual frontend domain before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vectorizer = None


@app.on_event("startup")
def load_artifacts():
    """Load model + vectorizer once at startup instead of on every request."""
    global model, vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Could not load model artifacts: {e}")
        # Leave model/vectorizer as None; /predict will return a 503 until fixed.


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


class PredictionResponse(BaseModel):
    sentiment: str
    input_text: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None and vectorizer is not None else "model_not_loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(input: TextInput):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    cleaned = clean_text(text)
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Input text has no usable content after cleaning",
        )

    try:
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictionResponse(sentiment=str(prediction), input_text=text)
