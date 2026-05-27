import joblib
import numpy as np

# Load saved model
model = joblib.load("D:\\Esiot_project\\ANUSHRI\\soil_model.pkl")


# Sample input:
# N, P, K, temperature, humidity, ph, rainfall
sample = np.array([[90, 40, 40, 25, 80, 6.5, 200]])

# Predict
prediction = model.predict(sample)

print("Predicted Soil Condition:", prediction[0])