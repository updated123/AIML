import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_clustering(df: pd.DataFrame):
    if "Salary" not in df.columns:
        return {"error": "Missing 'Salary' column"}

    X = df[["Salary"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    df["Cluster"] = clusters
    summary = df.groupby("Cluster")["Salary"].agg(["mean", "count"]).to_dict()

    return {"clusters": clusters.tolist(), "cluster_summary": summary}
