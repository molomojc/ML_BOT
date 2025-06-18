from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)

def safe_download(ticker, period, interval):
    """Safe data download with error handling"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None, "No data returned"
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.columns = [str(col).lower() for col in data.columns]
        return data, None
    except Exception as e:
        return None, str(e)

def generate_signal():
    """Generate trading signal with proper error handling"""
    data, error = safe_download("EURUSD=X", period="8d", interval="1m")
    if error:
        return {"error": error}, 500
    
    try:
        # Feature Engineering
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(4).std()
        data['range'] = (data['high'] - data['low']) / data['close'].replace(0, 1e-10)
        
        # Lagged features
        for lag in [1, 2, 3]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
        
        data.dropna(inplace=True)
        
        if len(data) < 20:
            return {"error": "Insufficient data after processing"}, 400
        
        # Model Training
        X = data[['return_lag_1', 'return_lag_2', 'volatility', 'range']]
        y = (data['returns'].shift(-1) > 0).astype(int)
        
        model = LinearRegression()
        model.fit(X[:-1], y[:-1])
        
        # Generate signal
        latest_features = X.iloc[-1:].values
        prediction = model.predict(latest_features)[0]
        
        signal = {
            "symbol": "EURUSD",
            "signal": "BUY" if prediction > 0.55 else "SELL" if prediction < 0.45 else "HOLD",
            "confidence": float(abs(prediction - 0.5) * 2),
            "timestamp": datetime.utcnow().isoformat(),
            "price": float(data['close'].iloc[-1])
        }
        return signal
    except Exception as e:
        return {"error": f"Signal generation failed: {str(e)}"}, 500

@app.route('/signal')
def signal_endpoint():
    result = generate_signal()
    if isinstance(result, tuple):  # Error case
        response, status_code = result
        return jsonify(response), status_code
    return jsonify(result)  # Success case

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Turn off debug in production