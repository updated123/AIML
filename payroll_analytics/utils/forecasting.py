import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_payroll(df: pd.DataFrame):
    if "Effective Date" not in df.columns or "Salary" not in df.columns:
        return {"error": "Missing required columns"}

    df["Effective Date"] = pd.to_datetime(df["Effective Date"])
    df.set_index("Effective Date", inplace=True)
    df = df.resample("M").mean()

    model = ARIMA(df["Salary"].dropna(), order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=3)

    return {"forecast": forecast.tolist(), "summary": model_fit.summary().as_text()}
