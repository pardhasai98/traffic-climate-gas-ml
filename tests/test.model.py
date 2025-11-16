import pandas as pd
from src.preprocessing import clean_data
from src.model import train_model
from src.predict import predict

def test_train_and_predict():
    # Sample data
    df = pd.DataFrame({
        'traffic_density': [30, 45, 40, 20, 35],
        'temperature': [22, 23, 25, 20, 21],
        'humidity': [60, 58, 55, 65, 63],
        'price_per_litre': [3.5, 3.55, 3.6, 3.45, 3.5]
    })

    # Clean data
    df = clean_data(df)

    # Features and target
    X = df[['traffic_density', 'temperature', 'humidity']]
    y = df['price_per_litre']

    # Train model
    model = train_model(X, y)

    # Make predictions
    preds = predict(model, X)

    assert len(preds) == len(y)
    print("Test passed: train and predict works.")

if __name__ == "__main__":
    test_train_and_predict()
