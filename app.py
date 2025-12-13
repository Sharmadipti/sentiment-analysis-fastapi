import joblib
from pydantic import BaseModel
from fastapi import FastAPI
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")



app = FastAPI()


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}

class TextInput(BaseModel):
    text: str


from fastapi import HTTPException

@app.post("/predict")
def predict_sentiment(input: TextInput):

    text = input.text

    if not text or text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty"
        )

    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)[0]

    return {"sentiment": prediction}
