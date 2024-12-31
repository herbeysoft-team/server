import sys
import json
import joblib
import pandas as pd
# Load the model and label encoder
model = joblib.load('quiz_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

try:
    input_data = json.loads(sys.argv[1])  # Parse input
    print("Parsed input:", input_data, flush=True)

    input_df = pd.DataFrame([input_data], columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    prediction = model.predict(input_df)
    decoded_prediction = label_encoder.inverse_transform(prediction)
    print(decoded_prediction[0], flush=True)
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}", flush=True)
    sys.exit(1)


