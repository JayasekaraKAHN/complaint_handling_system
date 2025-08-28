from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def auto_cluster_categories(complaints, n_clusters=5):
    """
    Automatically groups complaints into n_clusters using TF-IDF + KMeans.

    Args:
        complaints (list or pd.Series): List of complaint texts.
        n_clusters (int): Number of clusters to generate.

    Returns:
        labels (list): Cluster labels for each complaint.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(complaints)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return labels
