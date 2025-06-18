from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import requests
import ta

app = Flask(__name__)


#This computes the indicators and sends a json to the post
# Endpoint to compute indicators and call prediction API
@app.route('/calculate-and-predict', methods=['GET'])
def calculate_and_predict():
    try:
        # 1. Get recent BTC-USD data
        df = yf.download('BTC-USD', period='2d', interval='1h')
        df.dropna(inplace=True)

        # 2. Compute indicators (latest row will be used)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['Volume_Oscillator'] = (df['Volume'].pct_change() * 100).fillna(0)

        # Example support/resistance logic (can improve)
        df['Support_Level'] = df['Low'].rolling(window=5).min()
        df['Resistance_Level'] = df['High'].rolling(window=5).max()

        # Fibonacci retracement (using recent high-low)
        recent_high = df['High'][-10:].max()
        recent_low = df['Low'][-10:].min()
        df['Fib_Retracement'] = recent_high - 0.618 * (recent_high - recent_low)

        # Latest values
        last_row = df.iloc[-1]
        indicators = {
            "Last_Close": last_row['Close'],
            "SMA_5": last_row['SMA_5'],
            "SMA_20": last_row['SMA_20'],
            "RSI": last_row['RSI'],
            "MACD": last_row['MACD'],
            "Signal": last_row['Signal'],
            "Volume_Oscillator": last_row['Volume_Oscillator'],
            "Support_Level": last_row['Support_Level'],
            "Resistance_Level": last_row['Resistance_Level'],
            "Fib_Retracement": last_row['Fib_Retracement']
        }

        # 3. Send to prediction REST API
        prediction_api_url = 'http://localhost:5000/predict-next-close'
        response = requests.post(prediction_api_url, json=indicators)

        # Return response
        return jsonify({
            "indicators_sent": indicators,
            "prediction_response": response.json()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=False)
