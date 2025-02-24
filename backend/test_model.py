from model_predict import predict_stock_movement

# Sample stock data (closing prices for a stock over 8 days)
sample_stock_data = [108, 107, 106, 105, 103, 102] 

# Run prediction function and print result
prediction = predict_stock_movement(sample_stock_data)
print(f"Predicted Movement: {prediction}")  # Should print "Up" or "Down"
