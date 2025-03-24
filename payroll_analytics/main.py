from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from utils.data_loader import load_csv
from utils.clustering import perform_clustering
from utils.forecasting import forecast_payroll
from utils.churn import predict_churn
from utils.anomaly import detect_anomalies

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Request model
class RequestPayload(BaseModel):
    file_path: str
    task_name: str

@app.post("/process")
def process_data(payload: RequestPayload):
    file_path = payload.file_path
    task_name = payload.task_name.lower()

    # Load CSV file
    df = load_csv(file_path)
    if df is None:
        raise HTTPException(status_code=400, detail="File not found or invalid")

    # Perform task
    if task_name == "clustering":
        result = perform_clustering(df)
    elif task_name == "timeseries_forecasting":
        result = forecast_payroll(df)
    elif task_name == "churn_prediction":
        result = predict_churn(df)
    elif task_name == "anomaly_detection":
        result = detect_anomalies(df)
    else:
        raise HTTPException(status_code=400, detail="Invalid task_name")

    return {"status": "success", "task": task_name, "results": result}
