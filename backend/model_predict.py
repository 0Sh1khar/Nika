import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# ✅ Auto-detect device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define LSTM Model (Matching Saved Model)
class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# ✅ Load Model (With Error Handling)
model_path = os.path.join(os.path.dirname(__file__), "models/lstm_model.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model file not found! Train and save the model first.")

model = LSTM().to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ LSTM Model Loaded Successfully.")
except RuntimeError as e:
    print(f"❌ Model Loading Error: {e}")
    raise SystemExit("⚠️ Model architecture mismatch! Use the correct architecture.")

# Instead of loading a scaler from file, we now re-fit the scaler on the current 'Close' column.
# This scaler is fit on the incoming stock data only.
def predict_stock_movement(stock_data):
    """
    Predicts stock movement using the LSTM model.
    Expects stock_data as a Pandas DataFrame with a 'Close' column.
    Re-fits a MinMaxScaler on the 'Close' column only, then scales the data,
    passes it through the model, inverse-transforms the prediction, and compares
    it with the latest actual close price. Returns "Up" if the predicted price
    is higher than the latest close price; otherwise "Down".
    """
    print(f"\n🔹 Raw Data Received (tail):\n{stock_data.tail()}")

    # Extract the 'Close' prices and ensure they are floats.
    close_data = stock_data["Close"].values.reshape(-1, 1).astype(np.float32)
    if np.isnan(close_data).any():
        raise ValueError("❌ Error: Stock data contains NaN values!")

    # Re-fit a new scaler on only the 'Close' column.
    new_scaler = MinMaxScaler(feature_range=(-1, 1))
    new_scaler.fit(close_data)
    scaled_data = new_scaler.transform(close_data)
    print(f"🔹 Scaled Data (last 5):\n{scaled_data[-5:]}")

    # Extract the last scaled close value for later comparison.
    last_close_scaled = scaled_data[-1, 0]

    # Convert the entire scaled sequence to a PyTorch tensor and reshape for the LSTM: (1, sequence_length, 1)
    x_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Get the raw prediction from the model (normalized prediction)
    with torch.no_grad():
        raw_prediction = model(x_input).cpu().numpy()  # Shape (1, 1)
    print(f"🟢 Raw Normalized Model Prediction: {raw_prediction}")

    # Inverse transform the normalized prediction to get the predicted price in the original scale.
    predicted_price = new_scaler.inverse_transform(raw_prediction)[0, 0]
    # Inverse transform the last scaled close to get the latest actual close price.
    last_close_price = new_scaler.inverse_transform(np.array([[last_close_scaled]], dtype=np.float32))[0, 0]
    
    print(f"🟢 Inverse Transformed Predicted Price: {predicted_price}")
    print(f"📊 Latest Close Price: {last_close_price}")
    
    # Determine movement: "Up" if predicted price > latest close price, else "Down"
    movement = "Up" if predicted_price > last_close_price else "Down"
    print(f"🚀 Predicted Movement: {movement}")
    
    return movement
