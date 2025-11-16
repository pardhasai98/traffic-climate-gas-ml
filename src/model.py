import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model_from_csv(processed_csv="data/processed/processed_data.csv", save_model_path="model.pkl"):
    df = pd.read_csv(processed_csv)
    X = df[['traffic_density', 'temperature', 'humidity']]
    y = df['price_per_litre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, save_model_path)
    print(f"Model trained and saved to {save_model_path}")
    return model, X_test, y_test

if __name__ == "__main__":
    train_model_from_csv()
