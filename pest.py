import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Assuming you have a dataset containing historical data on crop diseases and pest infestations
crop_data = pd.read_csv('Semicolon Pest.csv')

# Preprocess the data
# Example: Extract relevant features such as temperature, humidity, soil moisture, etc.
X = crop_data[['Temperature (°C)', 'Humidity (%)', 'Soil_Moisture (%)']]
y = crop_data['Pest_Disease_Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

# Save the trained model as a pickle file
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict pest or disease outbreaks
new_data = np.array([[30, 60, 50]])  # Example new data for temperature, humidity, soil moisture
prediction = model.predict(new_data)
print(f'Predicted Pest or Disease Type: {prediction[0]}')

# Custom function to provide recommendations based on predicted classes
def provide_recommendations(predicted_class):
    recommendations = {
        'Blast': 'Crop Rotation: Rotate to non-host crops such as legumes or grasses to disrupt the disease cycle.\n'
                 'Pesticides: Fungicides containing active ingredients such as azoxystrobin, trifloxystrobin, '
                 'or propiconazole can be effective against blast.',
        'Downy Mildew': 'Crop Rotation: Rotate to non-host crops to reduce pathogen buildup in the soil.\n'
                        'Pesticides: Fungicides containing active ingredients such as metalaxyl, mancozeb,'
                        'or mefenoxam can help manage downy mildew.',
        'Grasshoppers':'Crop Rotation: Grasshoppers are less likely to infest diverse crops, so rotating to different crops can help reduce their population.\n'
                       'Pesticides: Insecticides containing active ingredients such as spinosad, cyfluthrin,'
                       'or carbaryl can be effective against grasshoppers.',
        'Birds':'Crop Rotation: Not Applicable/n'
                'Pesticides: There are no specific pesticides for bird control. Non-lethal deterrents such as scarecrows, netting,'
                'or noise devices may be used to deter birds.',
        'Leafhoppers':'Crop Rotation: Rotate to non-host crops to disrupt the pests life cycle./n'
                      'Pesticides: Insecticides containing active ingredients such as imidacloprid, thiamethoxam,'
                       'or dinotefuran can be effective against leafhoppers.',
        'Earworms':'Crop Rotation: Not applicable.\n'
                   'Pesticides: Insecticides containing active ingredients such as chlorantraniliprole, spinosad,'
                   'or Bacillus thuringiensis (Bt) can be effective against earworms.',
        'Aphids':'Crop Rotation: Rotate to non-host crops to reduce aphid populations.\n'
                 'Pesticides: Insecticides containing active ingredients such as imidacloprid, acetamiprid,'
                 'or pyrethroids can be effective against aphids.',
        'Stem Borers':'Crop Rotation: Rotate to non-host crops to disrupt the pests life cycle.\n'
                      'Pesticides: Insecticides containing active ingredients such as chlorpyrifos, cypermethrin,'
                      'or lambda-cyhalothrin can be effective against stem borers.'
              
    }
    return recommendations.get(predicted_class, 'No specific recommendations available for this pest/disease.')

# Example usage: Provide recommendations for the first predicted class
recommendations = provide_recommendations(prediction[0])
print(f'Recommendations for the {prediction[0]}:', recommendations)
