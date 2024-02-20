from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open('weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route to render the HTML page
@app.route('/')
def index():
    return render_template('tstweb.html')

# Define a route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    today_temperature = float(data['temperature'])
    today_humidity = float(data['humidity'])
    today_precipitation = float(data['precipitation'])

    # Predict future weather for the next 3 days
    future_weather_predictions = predict_future_weather(today_temperature, today_humidity, today_precipitation)

    # Format the prediction results
    predictions_formatted = []
    for i, prediction in enumerate(future_weather_predictions, start=1):
        predictions_formatted.append({
            'day': f'Day {i}',
            'temperature': f"{prediction[0]}°C",
            'humidity': f"{prediction[1]}%",
            'precipitation': f"{prediction[2]}mm"
        })

    # Return the prediction results as JSON response
    return jsonify(predictions_formatted)

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

if __name__ == '__main__':
    app.run(debug=True)
