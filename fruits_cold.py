# fruits_cold_chain_full.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# Step 1: Load dataset
# -------------------------------
fruits_df = pd.read_csv(r"C:\Users\Prajjit Basu\OneDrive\Desktop\python_proj\Dataset.csv")
print("✅ Fruits dataset loaded")

# -------------------------------
# Step 2: Drop unwanted columns
# -------------------------------
columns_to_drop = ["Sno", "Days"]  # keep CO2 this time
fruits_df = fruits_df.drop(columns=[c for c in columns_to_drop if c in fruits_df.columns])

# -------------------------------
# Step 3: Handle missing values
# -------------------------------
fruits_df = fruits_df.ffill()

# -------------------------------
# Step 4: Encode target variable
# -------------------------------
fruits_df['Spoiled'] = fruits_df['Spoiled'].map({'No': 1, 'Yes': 0})
target_column = "Spoiled"

# -------------------------------
# Step 5: Encode Fruit column
# -------------------------------
if fruits_df['Fruit'].dtype == 'object':
    le = LabelEncoder()
    fruits_df['Fruit'] = le.fit_transform(fruits_df['Fruit'])
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/fruit_label_encoder.pkl")

# -------------------------------
# Step 6: Scale numeric columns
# -------------------------------
feature_cols = [c for c in fruits_df.columns if c != target_column]
numeric_cols = [c for c in feature_cols if c != 'Fruit']
scaler = StandardScaler()
fruits_df[numeric_cols] = scaler.fit_transform(fruits_df[numeric_cols])

# Save preprocessed dataset
os.makedirs("processed_data", exist_ok=True)
fruits_df.to_csv("processed_data/fruits_cleaned.csv", index=False)
print("✅ Fruits dataset preprocessed and saved")

# -------------------------------
# Step 7: Train ML Model
# -------------------------------
X = fruits_df[feature_cols]
y = fruits_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save trained model + scaler
joblib.dump(model, "models/fruit_edible_model.pkl")
joblib.dump(scaler, "models/fruit_scaler.pkl")
print("✅ Fruit model trained and saved successfully")

# -------------------------------
# Step 8: Predict for every fruit
# -------------------------------
# Reload objects
model = joblib.load("models/fruit_edible_model.pkl")
scaler = joblib.load("models/fruit_scaler.pkl")
le = joblib.load("models/fruit_label_encoder.pkl")

# User input
temp = float(input("🌡️ Enter Temperature: "))
humidity = float(input("💧 Enter Humidity: "))
co2 = float(input("🟢 Enter CO₂ Level: "))

# Scale user input
user_input = pd.DataFrame([[temp, humidity, co2]], columns=["Temp", "Humidity", "CO2"])
user_input_scaled = scaler.transform(user_input)

# Predict for all fruits
results = []

for fruit_code, fruit_name in enumerate(le.classes_):
    input_with_fruit = pd.DataFrame(
        np.hstack([[[fruit_code]], user_input_scaled]),
        columns=["Fruit", "Temp", "Humidity", "CO2"]
    )
    prob = model.predict_proba(input_with_fruit)[0][1]  # edible=1
    results.append((fruit_name, prob))

# Sort by highest probability
results.sort(key=lambda x: x[1], reverse=True)

print("\n🍎 Fruit-wise Edibility Probabilities (based on your input):\n")
for fruit_name, prob in results:
    print(f"➡️ {fruit_name}: {prob*100:.2f}% edible")
