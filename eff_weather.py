import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the data from the CSV file
data = pd.read_csv('Weatherdata.csv')

# Drop any rows with missing values
data.dropna(inplace=True)

# Encode the target variable (weather) using LabelEncoder
encoder = LabelEncoder()
data['Weather'] = encoder.fit_transform(data['Weather'])

# Split the data into features and target variable
X = data[['MinTemp', 'MaxTemp', 'AvgTemp', 'Precipitation', 'WindDirection', 'WindSpeed', 'Pressure']]
y = data['Weather']

# Encode categorical variables (e.g., WindDirection) using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # Output layer with 7 neurons for 7 weather categories
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Function to predict weather for the next 3 days
def predict_next_3_days_weather(today_temp, today_precipitation):
    # Assume new_data contains the features for the next 3 days
    new_data = np.array([[today_temp, today_temp, today_temp, today_precipitation, 0, 0, 0]])  # Only today's temperature and precipitation are used
    
    # Standardize the features
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions for the next 3 days
    predictions = model.predict(new_data_scaled)
    predicted_weather = np.argmax(predictions, axis=1)
    
    return predicted_weather

# Example usage:
today_temp = 35.0  # Today's temperature
today_precipitation = 0.0  # Today's precipitation
predicted_weather = predict_next_3_days_weather(today_temp, today_precipitation)

# Define a dictionary to map integer predictions to weather labels
weather_labels = {0: 'Clear', 1: 'Partly Cloudy', 2: 'Cloudy', 3: 'Rainy', 4: 'Stormy', 5: 'Windy', 6: 'Foggy'}

# Map integer predictions to weather labels
predicted_weather_labels = [weather_labels[p] for p in predicted_weather]

# Print predicted weather labels for the next 3 days
print(f'Predicted weather for the next 3 days: {predicted_weather_labels}')
