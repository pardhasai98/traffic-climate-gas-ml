import pandas as pd

def clean_data(df):
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.reset_index(drop=True)
    print(f"Data cleaned, shape: {df.shape}")
    return df
