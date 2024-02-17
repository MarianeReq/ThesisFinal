import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
# wip pa muna
from torch.autograd import Variable 

# Load historical stock data from Yahoo Finance 1 stock lang sa for testingg 
symbol = 'AYAAF'
df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2=9999999999&interval=1d&events=history')
print (df)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate technical indicators using the 'ta' library
df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume')

# Select relevant features and drop NaN values 
# invalid data consists of empty and incomplete rows and cols
# Select relevant features and drop NaN values 
selected_columns = ['Date', 'Close', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
                    'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap', 'volatility_atr', 
                    'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli',
                    'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli',
                    'volatility_dch', 'volatility_dcl', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                    'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx', 'trend_adx_pos',
                    'trend_adx_neg', 'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_trix', 
                    'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 
                    'trend_ichimoku_conv', 'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 
                    'trend_aroon_ind', 'momentum_rsi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 
                    'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_roc', 'others_dr', 
                    'others_dlr', 'others_cr']
df = df[selected_columns].dropna()
print (df)


if df.shape[0] == 0:
    print("No samples left after dropping columns. Please check your data.")
else:
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.drop(['Date', 'Close'], axis=1).values)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.drop(['Date', 'Close'], axis=1).values)

# Prepare the input data for the LSTM model
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length, 0]  # Assuming the 'Close' column is the target
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

sequence_length = 10  # Adjust as needed
X, y = create_sequences(scaled_data, sequence_length)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Set hyperparameters
input_size = X.shape[2]
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Instantiate the model
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model no TIs pa 
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()
with torch.no_grad():
    future_sequence = torch.from_numpy(scaled_data[-sequence_length:]).unsqueeze(0).float()
    future_prediction = model(future_sequence)

# Inverse transform the prediction
future_prediction = scaler.inverse_transform(np.array([[future_prediction.item()]]))

print(f'Predicted Stock Price: {future_prediction.item()}')
