from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle estimation requests
@app.route('/estimate', methods=['POST'])
def estimate():
    # Get data from the POST request
    data = request.form
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    soil_moisture = float(data['soil_moisture'])

    # Perform estimation using the model
    estimated_pest, recommendations = predict_pest(temperature, humidity, soil_moisture)

    # Return estimation results as JSON response
    return jsonify({'pest': estimated_pest, 'recommendations': recommendations})

# Function to predict pest and provide recommendations
def predict_pest(temperature, humidity, soil_moisture):
    # Make prediction using the model
    new_data = np.array([[temperature, humidity, soil_moisture]])
    prediction = model.predict(new_data)[0]

    # Provide recommendations based on the predicted class
    recommendations = provide_recommendations(prediction)

    return prediction, recommendations

# Custom function to provide recommendations based on predicted class
def provide_recommendations(predicted_class):
    recommendations = {
        'Blast': 'Crop Rotation: Rotate to non-host crops such as legumes or grasses to disrupt the disease cycle. Pesticides: Fungicides containing active ingredients such as azoxystrobin, trifloxystrobin, or propiconazole can be effective against blast.',
        'Downy Mildew': 'Crop Rotation: Rotate to non-host crops to reduce pathogen buildup in the soil. Pesticides: Fungicides containing active ingredients such as metalaxyl, mancozeb, or mefenoxam can help manage downy mildew.',
        'Grasshoppers': 'Crop Rotation: Grasshoppers are less likely to infest diverse crops, so rotating to different crops can help reduce their population. Pesticides: Insecticides containing active ingredients such as spinosad, cyfluthrin, or carbaryl can be effective against grasshoppers.',
        'Birds': 'Crop Rotation: Not Applicable. Pesticides: There are no specific pesticides for bird control. Non-lethal deterrents such as scarecrows, netting, or noise devices may be used to deter birds.',
        'Leafhoppers': 'Crop Rotation: Rotate to non-host crops to disrupt the pests life cycle. Pesticides: Insecticides containing active ingredients such as imidacloprid, thiamethoxam, or dinotefuran can be effective against leafhoppers.',
        'Earworms': 'Crop Rotation: Not applicable. Pesticides: Insecticides containing active ingredients such as chlorantraniliprole, spinosad, or Bacillus thuringiensis (Bt) can be effective against earworms.',
        'Aphids': 'Crop Rotation: Rotate to non-host crops to reduce aphid populations. Pesticides: Insecticides containing active ingredients such as imidacloprid, acetamiprid, or pyrethroids can be effective against aphids.',
        'Stem Borers': 'Crop Rotation: Rotate to non-host crops to disrupt the pests life cycle. Pesticides: Insecticides containing active ingredients such as chlorpyrifos, cypermethrin, or lambda-cyhalothrin can be effective against stem borers.'
    }
    return recommendations.get(predicted_class, 'No specific recommendations available for this pest/disease.')

if __name__ == '__main__':
    app.run(debug=True)
