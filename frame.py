import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel

# Download historical data from Yahoo Finance
stock_data = yf.download('AYAAY', start='2014-01-01', end='2024-01-01')

# Calculate indicators
sma = SMAIndicator(close=stock_data['Close'], window=20).sma_indicator()
ema = EMAIndicator(close=stock_data['Close'], window=20).ema_indicator()
macd = MACD(close=stock_data['Close']).macd()
adx = ADXIndicator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close']).adx()
rsi = RSIIndicator(close=stock_data['Close']).rsi()
stoch = StochasticOscillator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close']).stoch()
williams_r = WilliamsRIndicator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close']).williams_r()
obv = OnBalanceVolumeIndicator(close=stock_data['Close'], volume=stock_data['Volume']).on_balance_volume()
acc_dist = AccDistIndexIndicator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], volume=stock_data['Volume']).acc_dist_index()
atr = AverageTrueRange(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close']).average_true_range()
bb = BollingerBands(close=stock_data['Close']).bollinger_hband()
kc = KeltnerChannel(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close']).keltner_channel_hband()

# Combine indicators into a DataFrame
features = pd.DataFrame({
    'SMA': sma,
    'EMA': ema,
    'MACD': macd,
    'ADX': adx,
    'RSI': rsi,
    'Stochastic': stoch,
    'Williams_R': williams_r,
    'OBV': obv,
    'Acc/Dist': acc_dist,
    'ATR': atr,
    'BB_High': bb,
    'KC_High': kc
}, index=stock_data.index)

# Shift the target variable (next day's movement)
target = np.sign(stock_data['Close'].shift(-1) - stock_data['Close'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features[:-1], target[:-1], test_size=0.2, random_state=42)

# Handling Class Imbalance
oversampler = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Reshape features for LSTM input
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='tanh'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train_lstm, y_train_balanced, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test), callbacks=[early_stopping])

# Evaluate the model
_, accuracy = model.evaluate(X_test_lstm, y_test)

print("Accuracy:", accuracy)

# Predictions
predictions = model.predict(X_test_lstm)
predicted_classes = np.sign(predictions).astype("int32")

# Input the predicted result
predicted_result = np.sign(model.predict(X_test_lstm[-1:])[0][0]).astype("int32")
if predicted_result == 1:
    print("Predicted Result: Upward Trend")
else:
    print("Predicted Result: Downward Trend")
