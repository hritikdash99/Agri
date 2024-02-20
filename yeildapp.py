from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model and scaler
with open('yield_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

scaler = MinMaxScaler()

# Define a function to preprocess input data
def preprocess_input(data):
    mapping = {'high': 0, 'moderate': 1, 'low': 2, 'poorly': 0, 'partially': 1, 'adequately': 2, 'insufficient': 0, 'adequate': 1, 'poor': 0, 'average': 1, 'healthy': 2}
    data.replace(mapping, inplace=True)
    data['Yield Percentage'] = data['Yield Percentage'].str.rstrip('%').astype(float)
    return data

@app.route('/')
def index():
    return render_template('precision.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        input_data = pd.DataFrame(data, index=[0])
        input_data = preprocess_input(input_data)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        return render_template('precision.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
