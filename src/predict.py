import pandas as pd
import joblib

def predict_from_model(model_path="model.pkl", input_csv="data/processed/processed_data.csv"):
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    X = df[['traffic_density', 'temperature', 'humidity']]
    df['predicted_gas_price'] = model.predict(X)
    df.to_csv("data/processed/predictions.csv", index=False)
    print("Predictions saved to data/processed/predictions.csv")
    return df

if __name__ == "__main__":
    predict_from_model()
