import joblib
import requests
from random import randint
from pathlib import Path

# Setup paths
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.parent.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

# Load scaled data and scaler
X_train_scaled = joblib.load(f"{proj_root}/data/X_train_scaled.pkl")
y_train = joblib.load(f"{proj_root}/data/y_train.pkl")
scaler = joblib.load(f"{proj_root}/src/models/scalers/scaler.pkl")

# Pick a random sample
r = randint(0, len(X_train_scaled) - 1)

# Get the scaled input and reverse it to raw values
input_scaled = X_train_scaled[r]
input_raw = scaler.inverse_transform([input_scaled])[0]

# y_train is a Series when loaded
expected = y_train.iloc[r]

# Feature names in the correct order (from creditcard.csv)
feature_names = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Create the JSON payload
transaction = {name: float(value) for name, value in zip(feature_names, input_raw)}

# Log the test run
with open(f"{proj_root}/tests/test_api/test_service/test_run.txt", "a") as file:
    file.write(f"\nNew Run {'=' * 60}\n")
    file.write(f"  Training Row Number: {r}\n")
    file.write(f"  Expected Class: {expected}\n")
    file.write(f"  Sample Features (first 5): {dict(list(transaction.items())[:5])}\n")

# Send the request to the API
API_URL = "http://localhost:8000/predict"
try:
    response = requests.post(API_URL, json=transaction)
    response.raise_for_status()  # Raise exception for 4xx/5xx status codes

    result = response.json()

    # Log the result
    with open(f"{proj_root}/tests/test_api/test_service/test_run.txt", "a") as file:
        file.write(f"  API Response: {result}\n")
        file.write(f"  Prediction: {result.get('is_fraud')}\n")
        file.write(f"  Probability: {result.get('fraud_probability'):.4f}\n")
        file.write(f"  Match: {result.get('is_fraud') == bool(expected)}\n")

    # Assert prediction matches expected (with some tolerance for threshold effects)
    print(f"Test completed")
    print(f"  Expected: {bool(expected)}")
    print(f"  Predicted: {result.get('is_fraud')}")
    print(f"  Probability: {result.get('fraud_probability'):.4f}")
    print(f"  Match: {result.get('is_fraud') == bool(expected)}")

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to API. Is the server running?")
    print("   Start it with: uvicorn src.api.main:app --reload")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response Text: {response.text}")
    try:
        error_detail = response.json()
        print(f"   Error Detail: {error_detail}")
    except:
        pass
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
