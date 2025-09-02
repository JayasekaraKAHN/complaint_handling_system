import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
import os
import pickle
import json
from typing import Dict, List, Optional

MODEL_PATH = "models/complaint_classifier.pkl"
PATTERNS_PATH = "models/patterns.json"

def load_retrained_model():
    """Load the retrained model and patterns"""
    try:
        # Load the classifier model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load patterns
        with open(PATTERNS_PATH, 'r') as f:
            patterns = json.load(f)
        
        return model, patterns
    except FileNotFoundError:
        print("⚠️ Retrained model not found, falling back to basic classification")
        return None, None

def classify_complaint_with_retrained_model(complaint_text: str) -> str:
    """Classify complaint using the retrained model"""
    model, patterns = load_retrained_model()
    
    if model is None:
        return classify_complaint(complaint_text)  # Fallback to old method
    
    try:
        # Use the retrained model for prediction
        prediction = model.predict([complaint_text])[0]
        return prediction
    except Exception as e:
        print(f"Error with retrained model: {e}")
        return classify_complaint(complaint_text)  # Fallback

def get_solution_from_patterns(complaint_text: str, device_info: str = "", site_info: str = "") -> Optional[str]:
    """Get solution using extracted patterns from retraining"""
    model, patterns = load_retrained_model()
    
    if patterns is None:
        return None
    
    # Combine text for matching
    combined_text = f"{complaint_text} {device_info} {site_info}".lower().strip()
    
    # Check exact mappings first
    solution_mappings = patterns.get('solution_mappings', {})
    if combined_text in solution_mappings:
        return solution_mappings[combined_text]
    
    # Check device patterns
    device_patterns = patterns.get('device_patterns', {})
    for device, solution in device_patterns.items():
        if device.lower() in combined_text:
            return solution
    
    # Check site patterns
    site_patterns = patterns.get('site_patterns', {})
    for site, solution in site_patterns.items():
        if site.lower() in combined_text:
            return solution
    
    return None

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
