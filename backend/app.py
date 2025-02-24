from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from data_fetcher import get_stock_data
from model_predict import predict_stock_movement
from sentiment import analyze_sentiment
from news_fetcher import fetch_stock_news

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

def format_price(price, symbol):
    """
    Formats the price with a single currency symbol.
    If the stock symbol ends with ".NS", uses the rupee symbol (₹); otherwise, uses the dollar symbol ($).
    """
    desired_symbol = "₹" if symbol.endswith(".NS") else "$"
    # Format the numeric price to a string with 2 decimals
    formatted_price = f"{price:.2f}"
    return f"{desired_symbol}{formatted_price}"

@app.route('/')
def serve_home():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        print(f"❌ Error serving home page: {e}")
        return jsonify({'error': 'Failed to load home page'}), 500

@app.route('/stock.html')
def serve_stock():
    try:
        return send_from_directory(app.static_folder, 'stock.html')
    except Exception as e:
        print(f"❌ Error serving stock page: {e}")
        return jsonify({'error': 'Failed to load stock page'}), 500

@app.route('/news.html')
def serve_news():
    try:
        return send_from_directory(app.static_folder, 'news.html')
    except Exception as e:
        print(f"❌ Error serving news page: {e}")
        return jsonify({'error': 'Failed to load news page'}), 500

@app.route('/api/stock', methods=['GET', 'POST'])
def fetch_stock():
    try:
        if request.method == 'POST':
            data = request.get_json()
            symbol = data.get('symbol', '').upper()
            period = data.get('period', '1mo')
        else:  # Handling GET requests for backward compatibility
            symbol = request.args.get('symbol', '').upper()
            period = request.args.get('period', '1mo')

        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400

        print(f"📊 Fetching stock data for: {symbol}, Period: {period}")
        stock_data = get_stock_data(symbol, period)

        if stock_data is None or len(stock_data) == 0:
            print(f"🚨 No stock data available for {symbol}")
            return jsonify({'error': 'Stock data not available'}), 404

        print(f"✅ Stock data retrieved: {stock_data.tail()}")

        # Use the latest valid prices from the DataFrame
        latest_price = stock_data["Close"].dropna().iloc[-1]
        open_price = stock_data["Open"].dropna().iloc[-1]
        high = stock_data["High"].dropna().iloc[-1]
        low = stock_data["Low"].dropna().iloc[-1]

        # Get the model prediction and convert it to "Up" or "Down"
        predicted_price = predict_stock_movement(stock_data)
        if isinstance(predicted_price, (int, float, np.number)):
            movement = "Up" if predicted_price > latest_price else "Down"
        else:
            movement = predicted_price

        sentiment = analyze_sentiment(symbol)

        volume = stock_data["Volume"].dropna().iloc[-1]
        volume = int(volume) if isinstance(volume, (int, float)) else 0

        # Reset the index so that the Date becomes a column in the history.
        history_with_date = stock_data.reset_index().to_dict(orient="records")

        response = {
            "symbol": symbol,
            "latest_price": format_price(float(latest_price), symbol),
            "open_price": format_price(float(open_price), symbol),
            "high": format_price(float(high), symbol),
            "low": format_price(float(low), symbol),
            "volume": volume,
            "prediction": movement,
            "sentiment": sentiment,
            "history": history_with_date,
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return jsonify({'error': f'Failed to fetch stock data. Try again later. ({str(e)})'}), 500

@app.route('/api/news', methods=['GET'])
def fetch_news():
    symbol = request.args.get('symbol', '').upper()
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400

    try:
        print(f"📰 Fetching news for: {symbol}")
        news_articles = fetch_stock_news(symbol)

        if not news_articles:
            print(f"🚨 No news found for {symbol}")
            return jsonify({'error': 'No news found'}), 404

        sentiment_analysis = analyze_sentiment(news_articles)

        return jsonify({
            'symbol': symbol,
            'news': news_articles,
            'sentiment': sentiment_analysis
        })

    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return jsonify({'error': 'Failed to fetch news. Try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
