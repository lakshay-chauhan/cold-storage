Implemented the supervised machine learning pipelines for predicting vaccine container temperatures and assessing fruit edibility based on environmental conditions. The work includes:

Vaccine Temperature Prediction (supervised_vaccine_predict.py)
    Preprocessed temperature and humidity data for modeling.
    Scaled features using StandardScaler.
    Built a Random Forest Regressor to predict thermal shipper temperatures.
    Integrated a real-time user input system to predict vaccine container temperature and display safety warnings.
    Saved the trained model and scaler for future predictions.
Fruit Cold Chain Monitoring (fruits_cold_chain_full.py)
    Preprocessed fruit dataset, handled missing values, and encoded categorical features.
    Built a Random Forest Classifier to predict fruit edibility.
    Implemented a feature scaling pipeline for temperature, humidity, and CO₂ readings.
    Created a fruit-wise probability prediction system based on user input.
    Saved trained models, scaler, and label encoder for deployment-ready use.

Key Technologies Used:
Python | pandas | NumPy | scikit-learn | joblib | Random Forest | StandardScaler | LabelEncoder

Impact:
Enables predictive monitoring for cold chain management.
Helps ensure vaccine safety and fruit quality, reducing waste and maintaining health standards.
