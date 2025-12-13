# Sentiment Analysis API using NLP and FastAPI

## Project Overview
This project is an end-to-end Sentiment Analysis system that predicts whether a given text review is positive or negative.
The machine learning model is trained using classical NLP techniques and deployed as a REST API using FastAPI.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- FastAPI
- Uvicorn
- Joblib
- Pydantic

---

## Machine Learning Workflow
1. Data cleaning and preprocessing
2. Text vectorization using TF-IDF
3. Model training using Logistic Regression
4. Model evaluation using accuracy, precision, recall, and F1-score
5. Saving trained model and vectorizer
6. Deploying the model as a REST API using FastAPI

---

## API Endpoints

### GET `/`
Health check endpoint to verify that the API is running.

Response:
```json
{
  "message": "Sentiment Analysis API is running"
}
