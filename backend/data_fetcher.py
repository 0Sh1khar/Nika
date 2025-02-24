import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

def get_stock_data(symbol, period="3mo", look_back=60):
    try:
        print(f"📊 Fetching stock data for {symbol} with period: {period}...")

        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)

        if hist.empty or "Close" not in hist:
            raise ValueError(f"❌ No closing price data available for {symbol}")

        # Convert "Close" column to numeric; if any non-numeric, coerce to NaN
        hist["Close"] = pd.to_numeric(hist["Close"], errors='coerce')
        # Forward-fill missing values so we don't end up with NaN or 0
        hist["Close"].fillna(method='ffill', inplace=True)

        # Ensure that the last close price is not zero
        if hist["Close"].iloc[-1] == 0:
            valid_prices = hist["Close"][hist["Close"] > 0]
            if not valid_prices.empty:
                hist["Close"].iloc[-1] = valid_prices.iloc[-1]
            else:
                raise ValueError(f"❌ All close prices are 0 for {symbol}")

        # Optionally, you might want to convert other numeric columns similarly
        for col in ["Open", "High", "Low", "Volume"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors='coerce')
                hist[col].fillna(method='ffill', inplace=True)

        # Auto-adjust look_back if not enough data
        if len(hist) < look_back:
            look_back = len(hist)

        if len(hist) == 0:
            raise ValueError(f"❌ Not enough historical data for {symbol} (needed {look_back}, got {len(hist)})")

        # For debugging: print last close price
        print(f"✅ Successfully fetched {len(hist)} days of data for {symbol}")
        print(f"📈 Last Close Price (Fixed): {hist['Close'].iloc[-1]}")

        # (Optional) If you need normalized data for your model, you can compute scaled_data:
        # close_prices = hist["Close"].values.reshape(-1, 1)
        # scaled_data = scaler.fit_transform(close_prices)
        # You can return either hist (original data) or scaled_data as needed.
        return hist

    except Exception as e:
        print(f"🚨 Error fetching stock data: {e}")
        return None

# Run a test fetch when the script is executed
if __name__ == "__main__":
    test_symbol = "AAPL"
    data = get_stock_data(test_symbol, "1mo")
    if data is not None:
        print(f"Sample data for {test_symbol}:")
        print(data.tail())
    else:
        print("❌ Failed to fetch stock data.")
