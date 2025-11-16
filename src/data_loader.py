import pandas as pd
import os
from src.preprocessing import clean_data

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_data(raw_data_path="data/raw/"):
    traffic = load_csv(os.path.join(raw_data_path, "traffic_data.csv"))
    climate = load_csv(os.path.join(raw_data_path, "climate_data.csv"))
    gas = load_csv(os.path.join(raw_data_path, "gas_prices.csv"))
    return {"traffic": traffic, "climate": climate, "gas": gas}

def merge_datasets(datasets):
    df = datasets["traffic"]
    df = df.merge(datasets["climate"], on=["timestamp", "location"], how="left")
    df = df.merge(datasets["gas"], on=["timestamp", "location"], how="left")
    print(f"Merged dataset shape: {df.shape}")
    return df

def save_processed(df, processed_path="data/processed/processed_data.csv"):
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved at {processed_path}")

if __name__ == "__main__":
    datasets = load_all_data()
    df = merge_datasets(datasets)
    df = clean_data(df)
    save_processed(df)
