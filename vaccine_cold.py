# supervised_vaccine_predict.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# -------------------------------
# Step 1: Load dataset
# -------------------------------
vaccine_df = pd.read_csv(r"C:\Users\Prajjit Basu\OneDrive\Desktop\python_proj\input_data.csv")
print("✅ Vaccine dataset loaded")

# Keep only relevant columns
columns_to_keep = ["thermal_shipper_temp_reading", "room_temp_reading", "room_humidity_reading"]
missing_cols = [c for c in columns_to_keep if c not in vaccine_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

vaccine_df = vaccine_df[columns_to_keep]

# -------------------------------
# Step 2: Handle missing values
# -------------------------------
vaccine_df = vaccine_df.ffill()

# -------------------------------
# Step 3: Scale features
# -------------------------------
target_column = "thermal_shipper_temp_reading"
feature_cols = [c for c in vaccine_df.columns if c != target_column]

scaler = StandardScaler()
vaccine_df[feature_cols] = scaler.fit_transform(vaccine_df[feature_cols])

# -------------------------------
# Step 4: Supervised Learning (Train Model)
# -------------------------------
X = vaccine_df[feature_cols]
y = vaccine_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("🌡️ Vaccine Temperature Prediction MSE:", mean_squared_error(y_test, y_pred))

# -------------------------------
# Step 5: Save model and scaler
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/vaccine_temp_model.pkl")
joblib.dump(scaler, "models/vaccine_scaler.pkl")
print("✅ Model and scaler saved successfully\n")

# -------------------------------
# Step 6: Take user input and predict
# -------------------------------
while True:
    print("Enter new room conditions to predict vaccine container temperature (type 'exit' to quit):")
    room_temp_input = input("Room Temperature (°C): ")
    if room_temp_input.lower() == "exit":
        break
    room_humidity_input = input("Room Humidity (%): ")
    if room_humidity_input.lower() == "exit":
        break
    
    try:
        room_temp = float(room_temp_input)
        room_humidity = float(room_humidity_input)
    except ValueError:
        print("⚠️ Invalid input. Please enter numeric values.\n")
        continue

    # Prepare and scale input
    input_df = pd.DataFrame({
        "room_temp_reading": [room_temp],
        "room_humidity_reading": [room_humidity]
    })
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # Predict container temperature
    predicted_temp = model.predict(input_scaled)[0]
    print(f"\nPredicted Vaccine Container Temperature: {predicted_temp:.2f}°C")

    # Display warning based on safe range
    if -2 <= predicted_temp <= 5:
        print("✅ Vaccine is in appropriate environment\n")
    elif predicted_temp < -2:
        print("⚠️ Beware: Increase the temperature\n")
    else:
        print("⚠️ Beware: Decrease the temperature\n")
