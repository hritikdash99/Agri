import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Assuming you have historical weather data stored in a CSV file
weather_data = pd.read_csv('Semicolon Ragi1.csv')

# Preprocess the data
# Example: Extract relevant features such as temperature, humidity, precipitation, etc.
X = weather_data[['Temperature', 'Humidity', 'Precipitation']]
y = weather_data[['Future_Temperature', 'Future_Humidity', 'Future_Precipitation']]

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model as a pickle file
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Function to predict future weather for the next 3 days
def predict_future_weather(today_temperature, today_humidity, today_precipitation):
    future_weather_predictions = []
    for i in range(3):  # Predict for the next 3 days
        # Predict future weather for the next day
        future_weather_prediction = model.predict([[today_temperature, today_humidity, today_precipitation]])
        future_weather_predictions.append(future_weather_prediction[0])
        # Update input data for the next prediction
        today_temperature, today_humidity, today_precipitation = future_weather_prediction[0]
    return future_weather_predictions

# Example input for today's weather
today_temperature = 21
today_humidity = 86
today_precipitation = 0

# Predict future weather for the next 3 days
future_weather_predictions = predict_future_weather(today_temperature, today_humidity, today_precipitation)
print("Future Weather Predictions (Next 3 Days):")
print(f"Day 1 - Temperature: {future_weather_predictions[0][0]}°C, Humidity: {future_weather_predictions[0][1]}%, Precipitation: {future_weather_predictions[0][2]}mm")
print(f"Day 2 - Temperature: {future_weather_predictions[1][0]}°C, Humidity: {future_weather_predictions[1][1]}%, Precipitation: {future_weather_predictions[1][2]}mm")
print(f"Day 3 - Temperature: {future_weather_predictions[2][0]}°C, Humidity: {future_weather_predictions[2][1]}%, Precipitation: {future_weather_predictions[2][2]}mm")
