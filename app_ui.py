import streamlit as st
import joblib
import pandas as pd

st.title("🏡 House Price Prediction")

# Load model directly
model = joblib.load("models/house_price_model.pkl")
feature_columns = model.feature_names_in_

overall_qual = st.slider("Overall Quality", 1, 10, 5)
area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage = st.slider("Garage Cars", 0, 4, 1)
basement = st.number_input("Basement Area", 0, 2000, 500)

if st.button("Predict Price"):

    # Create full feature vector
    data = [0] * len(feature_columns)
    df = pd.DataFrame([data], columns=feature_columns)

    # Fill selected features
    if "OverallQual" in df.columns:
        df["OverallQual"] = overall_qual
    if "GrLivArea" in df.columns:
        df["GrLivArea"] = area
    if "GarageCars" in df.columns:
        df["GarageCars"] = garage
    if "TotalBsmtSF" in df.columns:
        df["TotalBsmtSF"] = basement

    prediction = model.predict(df)[0]

    st.success(f"Predicted Price: ₹ {int(prediction):,}")