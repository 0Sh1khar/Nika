import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Ensure models directory exists
os.makedirs("backend/models", exist_ok=True)

# 🔹 Fetch stock data from Yahoo Finance
ticker = "AAPL"  # Change this to any stock symbol
df = yf.download(ticker, start="2015-01-01", end="2025-01-01")
df = df[['Close']].fillna(method='ffill')  # Use only the Close price

# 🔹 Normalize data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 🔹 Prepare training data
def load_data(stock, look_back=60):
    data_raw = stock.values
    data = [data_raw[i: i + look_back] for i in range(len(data_raw) - look_back)]
    data = np.array(data)

    train_size = int(len(data) * 0.8)
    
    x_train, y_train = data[:train_size, :-1], data[:train_size, -1]
    x_test, y_test = data[train_size:, :-1], data[train_size:, -1]

    # Reshape to fit LSTM input (batch_size, sequence_length, input_size)
    x_train = x_train.reshape(-1, look_back - 1, 1)
    x_test = x_test.reshape(-1, look_back - 1, 1)

    return torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_test), torch.Tensor(y_test)

x_train, y_train, x_test, y_test = load_data(df)

# 🔹 Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Take the last output from LSTM

model = LSTM()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 🔹 Train model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    y_train_pred = model(x_train)

    # Reshape y_train for compatibility
    y_train = y_train.view(-1, 1)

    loss = loss_fn(y_train_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 🔹 Save trained model
torch.save(model.state_dict(), "backend/models/lstm_model.pth")
print("✅ Model saved successfully at backend/models/lstm_model.pth")
