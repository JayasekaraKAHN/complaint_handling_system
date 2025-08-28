import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
import os

MODEL_PATH = "app/models/complaint_classifier.pkl"

def train_classifier(data_path: str):
    # Try UTF-8 first, fall back to ISO-8859-1 if it fails
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decoding failed, falling back to ISO-8859-1")
        df = pd.read_csv(data_path, encoding="ISO-8859-1")

    # Ensure complaint text column exists
    if "Issue Description" not in df.columns:
        raise ValueError("CSV must contain an 'Issue Description' column")

    complaint_col = "Issue Description"

    # If Category missing → auto-generate with clustering
    if "Category" not in df.columns:
        print("⚠️ No 'Category' column found. Auto-generating categories via clustering...")

        # Vectorize complaints
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_vec = vectorizer.fit_transform(df[complaint_col].astype(str))

        # Create 5 clusters (you can tune this)
        n_clusters = min(5, len(df))  # prevent crash if dataset < 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Category"] = kmeans.fit_predict(X_vec).astype(str)

    # Extract categories
    categories = df["Category"].unique().tolist()
    print(f"Detected complaint categories: {categories}")

    # Train classifier
    X = df[complaint_col].astype(str)
    y = df["Category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    acc = pipeline.score(X_test, y_test)
    print(f"✅ Classifier trained. Accuracy: {acc:.2f}")
    return pipeline, categories


def load_classifier():
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
    except Exception:
        pass
    return None


def classify_complaint(complaint_text: str) -> str:
    model = load_classifier()
    if not complaint_text:
        return "Unknown"
    if model is None:
        return "Unknown"
    try:
        return model.predict([complaint_text])[0]
    except Exception:
        return "Unknown"
