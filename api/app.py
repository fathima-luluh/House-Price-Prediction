from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load(r"C:\Users\fathi\Documents\House-Price-Prediction\models\house_price_model.pkl")
feature_columns = model.feature_names_in_

# -----------------------------
# Input Schema
# -----------------------------
class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float

# -----------------------------
# Output Schema
# -----------------------------
class PredictionOutput(BaseModel):
    predicted_price: float

# -----------------------------
# Home
# -----------------------------
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

# -----------------------------
# Predict
# -----------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: HouseInput):
    try:
        # Create full feature vector (all zeros)
        input_data = [0] * len(feature_columns)
        df = pd.DataFrame([input_data], columns=feature_columns)

        # Fill only selected features
        for col in df.columns:
            if col == "OverallQual":
                df[col] = data.OverallQual
            elif col == "GrLivArea":
                df[col] = data.GrLivArea
            elif col == "GarageCars":
                df[col] = data.GarageCars
            elif col == "TotalBsmtSF":
                df[col] = data.TotalBsmtSF

        prediction = model.predict(df)[0]

        return {"predicted_price": float(prediction)}

    except Exception as e:
        return {"predicted_price": 0.0}

print("Loading model...")
model = joblib.load("models/house_price_model.pkl")
print("Model loaded successfully!")