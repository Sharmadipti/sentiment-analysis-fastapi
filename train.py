"""
Trains the sentiment analysis model and saves it to disk.

Usage:
    python train.py

Reads data/reviews.csv (columns: text, label) and produces:
    sentiment_model.pkl
    tfidf_vectorizer.pkl

NOTE ON THE DATASET:
data/reviews.csv included here is a small synthetic template-based dataset,
meant to make this repo runnable end-to-end out of the box. For a real
portfolio-grade model, replace it with a real dataset such as the IMDB
50k movie review dataset (https://ai.stanford.edu/~amaas/data/sentiment/)
or the Amazon reviews dataset -- just keep the same two columns
(text, label) and this script will work unchanged.
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocessing import clean_text

DATA_PATH = "data/reviews.csv"
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print("Vectorizing text with TF-IDF ...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model ...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
