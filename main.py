# ===============================
# House Price Prediction (REAL DATA)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. LOAD DATA
# -------------------------------

df = pd.read_csv("data/housing.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. CLEANING
# -------------------------------

# Drop columns with too many missing values
df = df.dropna(thresh=len(df)*0.7, axis=1)

# Fill missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object', 'string']).columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -------------------------------
# 3. FEATURE SELECTION
# -------------------------------

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# Convert categorical → numeric
X = pd.get_dummies(X, drop_first=True)

print("Processed Shape:", X.shape)

# -------------------------------
# 4. TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. MODELS
# -------------------------------

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -------------------------------
# 6. PREDICTIONS
# -------------------------------

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# -------------------------------
# 7. EVALUATION
# -------------------------------

def evaluate(y_true, y_pred, name):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2:", r2_score(y_true, y_pred))

evaluate(y_test, lr_pred, "Linear Regression")
evaluate(y_test, rf_pred, "Random Forest")

# -------------------------------
# 8. VISUALIZATION
# -------------------------------

plt.figure()
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.savefig("images/real_prediction.png")

print("\nGraph saved in images folder")

import joblib
joblib.dump(rf, "models/house_price_model.pkl")

print("Model saved successfully!")