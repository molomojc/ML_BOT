from flask import Flask, request, jsonify
import yfinance as yf
from threading import Lock
import time

app = Flask(__name__)
lock = Lock()
latest_signal = None

def get_latest_btc_close():
    data = yf.download("EURUSD=X", period="1d", interval="1m")
    close_price = data["Close"].dropna().values.flatten()
    return float(close_price[-1]) if len(close_price) > 0 else None


@app.route('/receive_price', methods=['POST'])
def receive_price():
    global latest_signal
    try:
        data = request.get_json()
        mt_price = float(data["price"])
        symbol = data["symbol"]

        close_price = get_latest_btc_close()
        if close_price is None:
            return jsonify({"error": "Could not retrieve BTC close price."}), 500

        if close_price > mt_price:
            signal = "BUY"
            tp = close_price
        elif close_price < mt_price:
            signal = "SELL"
            tp = close_price
        else:
            signal = "HOLD"
            tp = mt_price

        with lock:
            latest_signal = {
                "symbol": symbol,
                "signal": signal,
                "entry_price": mt_price,
                "tp": round(tp, 2),
                "sl": round(mt_price * 0.99, 2) if signal == "BUY" else round(mt_price * 1.01, 2),
                "timestamp": time.time()
            }

        return jsonify(latest_signal)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_signal', methods=['GET'])
def get_signal():
    with lock:
        if latest_signal is None:
            return jsonify({"status": "no signal available"}), 404
        return jsonify(latest_signal)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)