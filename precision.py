import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Step 1: Load the dataset
dataset = pd.read_csv('Semicolon yeild.csv')

# Step 2: Prepare the data
# Encode categorical variables
mapping = {'high': 0, 'moderate': 1, 'low': 2, 'poorly': 0, 'partially': 1, 'adequately': 2, 'insufficient': 0, 'adequate': 1, 'poor': 0, 'average': 1, 'healthy': 2}
dataset.replace(mapping, inplace=True)

# Convert 'Yield Percentage' to numerical values
dataset['Yield Percentage'] = dataset['Yield Percentage'].str.rstrip('%').astype(float)

# Split the data into features and target
X = dataset.drop(columns=['Yield Percentage'])
y = dataset['Yield Percentage']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)

# Step 7: Save the trained model as a pickle file
with open('yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'yield_model.pkl'")
