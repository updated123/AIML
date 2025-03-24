import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def predict_churn(df: pd.DataFrame):
    required_columns = ["Date of Joining", "Last Working Date", "Status", "Employee Status", "Salary"]
    if not all(col in df.columns for col in required_columns):
        return {"error": "Missing required columns"}

    df["Date of Joining"] = pd.to_datetime(df["Date of Joining"])
    df["Last Working Date"] = pd.to_datetime(df["Last Working Date"], errors="coerce")

    df["Tenure"] = (df["Last Working Date"] - df["Date of Joining"]).dt.days.fillna(0)
    df["Churn"] = df["Status"].apply(lambda x: 1 if x == "Exited" else 0)

    X = df[["Tenure", "Salary"]].fillna(0)
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {"accuracy": accuracy, "predictions": y_pred.tolist()}
