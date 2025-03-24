import pandas as pd
import logging

def load_csv(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded file: {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        return None
