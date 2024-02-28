import yfinance as yf
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Download historical data from Yahoo Finance
stock_data = yf.download('AYAAY', start='2014-01-01', end='2024-01-01')

# Selecting relevant features
features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Scaling features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Define input and target data
X = scaled_features[:-1]  # Input features
y = scaled_features[1:, 0]  # Target variable (open price shifted by 1 day)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.50))
model.add(LSTM(units=500, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(units=500))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predicting the open price
predicted_open_price = model.predict(X_test[-1:])
print("Predicted Open Price:", predicted_open_price * (features['Open'].max() - features['Open'].min()) + features['Open'].min())

# Extracting the open prices for the last week
recent_open_prices = stock_data['Open']

# Plotting the recent open prices
plt.figure(figsize=(10, 6))
plt.plot(recent_open_prices.index, recent_open_prices.values, label='Recent Open Prices', marker='o')

# Plotting the predicted open price
predicted_open_price = 18.43  # Replace with your predicted value
plt.axhline(y=predicted_open_price, color='r', linestyle='--', label='Predicted Open Price')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Recent Open Prices and Predicted Open Price')
plt.legend()

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

# Displaying the plot
plt.grid(True)
plt.tight_layout()
plt.show()
