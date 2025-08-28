import pandas as pd
from . import config
from typing import List, Dict, Any
import numpy as np


def load_datasets():
    """
    Load usage and complaints datasets with fallback to empty DataFrames.
    """
    try:
        usage = pd.read_csv(config.USAGE_DATA)
    except Exception:
        usage = pd.DataFrame(columns=["MSISDN"])  # empty fallback

    try:
        complaints = pd.read_csv(config.COMPLAINTS_DATA, encoding="latin1")
    except Exception:
        complaints = pd.DataFrame(columns=["Impacted MSISDN", "Complaint", "Solution"])  # empty fallback

    return usage, complaints


def get_customer_context(msisdn: str) -> Dict[str, Any]:
    """
    Retrieve usage and complaints for a given MSISDN.
    """
    usage, complaints = load_datasets()

    usage_info = []
    comp_info = []
    if msisdn:
        if "MSISDN" in usage.columns:
            usage_info = usage[usage["MSISDN"].astype(str) == str(msisdn)].to_dict(orient="records")
        if "Impacted MSISDN" in complaints.columns:
            comp_info = complaints[complaints["Impacted MSISDN"].astype(str) == str(msisdn)].to_dict(orient="records")

    found = bool(usage_info) or bool(comp_info)

    return {
        "usage": usage_info,
        "complaints": comp_info,
        "found": found
    }


def retrieve_similar_solutions(query_text: str, k: int = 3) -> List[str]:
    """
    Return up to k solutions from the complaints dataset that are most similar to the query_text.
    If no query_text or dataset is empty, return up to k most frequent solutions.
    """
    _, complaints = load_datasets()
    if complaints.empty or "Complaint" not in complaints.columns:
        return []

    # Clean NA values
    cdf = complaints.copy()
    cdf["Complaint"] = cdf["Complaint"].fillna("")
    cdf["Solution"] = cdf.get("Solution", pd.Series([""] * len(cdf))).fillna("")

    # If query_text is empty, return most frequent solutions
    if not query_text.strip():
        freq = cdf["Solution"].value_counts()
        return [s for s in freq.index.tolist()[:k] if isinstance(s, str) and s.strip()]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        freq = cdf["Solution"].value_counts()
        return [s for s in freq.index.tolist()[:k] if isinstance(s, str) and s.strip()]

    # Vectorize complaints + query
    texts = cdf["Complaint"].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts + [query_text])  # sparse matrix

    # Extract query vector and compute similarity
    query_vec = X.getrow(len(texts)).toarray()      # Convert query vector to dense
    complaints_matrix = [X.getrow(i).toarray() for i in range(len(texts))]  # List of dense rows
    complaints_matrix = np.vstack(complaints_matrix)  # Stack into a 2D array
    sim = cosine_similarity(query_vec, complaints_matrix).flatten()

    # Get top k similar complaints
    top_idx = sim.argsort()[::-1][:k]
    solutions = []
    for idx in top_idx:
        sol = cdf.iloc[idx].get("Solution", "")
        if isinstance(sol, str) and sol.strip():
            solutions.append(sol.strip())

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for s in solutions:
        if s not in seen:
            seen.add(s)
            dedup.append(s)

    return dedup[:k]
