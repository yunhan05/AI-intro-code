import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# To turn off oneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset
data = pd.read_csv('pizza_sales.csv')
data['order_date'] = pd.to_datetime(data['order_date'], format='%d-%m-%Y')
data['day_of_week'] = data['order_date'].dt.dayofweek
features = data[['quantity', 'unit_price', 'total_price', 'day_of_week']]
target = data['total_price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model architecture
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Use Input layer to define input shape
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, batch_size=32)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print("Test MAE: ", test_mae)
