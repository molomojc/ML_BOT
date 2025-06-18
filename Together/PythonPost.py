from flask import Flask, request, jsonify
import pandas as pd
import joblib  # or pickle
import numpy as np

app = Flask(__name__)

# Load your trained regression model (adjust filename accordingly)
model = joblib.load("regression_model.pkl")  # Ensure this is your latest trained model

@app.route('/predict-next-close', methods=['POST'])
def predict_next_close():
    data = request.json

    try:
        # Extract indicators from the request
        indicators = pd.DataFrame([{
            'SMA_5': data['SMA_5'],
            'SMA_20': data['SMA_20'],
            'RSI': data['RSI'],
            'MACD': data['MACD'],
            'Signal': data['Signal'],
            'Volume_Oscillator': data['Volume_Oscillator'],
            'Support_Level': data['Support_Level'],
            'Resistance_Level': data['Resistance_Level'],
            'Fib_Retracement': data['Fib_Retracement']
        }])

        # Last close
        last_close = float(data['Last_Close'])

        # Predict next close
        predicted_close = model.predict(indicators)[0]

        return jsonify({
            "last_close": round(last_close, 2),
            "predicted_close": round(predicted_close, 2),
            "delta": round(predicted_close - last_close, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=False)
