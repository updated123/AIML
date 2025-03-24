import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame):
    if "Salary" not in df.columns:
        return {"error": "Missing 'Salary' column"}

    X = df[["Salary"]].dropna()
    model = IsolationForest(contamination=0.05)
    df["Anomaly"] = model.fit_predict(X)

    anomalies = df[df["Anomaly"] == -1]
    return {"anomalies": anomalies.to_dict(orient="records")}
