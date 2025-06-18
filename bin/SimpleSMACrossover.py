# trading_api.py
from flask import Flask, request, jsonify
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Configuration
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
PORT = 5000

class TradingEngine:
    def __init__(self):
        self.connected = False
        self.connect_mt5()
        
    def connect_mt5(self):
        if not mt5.initialize():
            print("MT5 Initialize Failed")
            return False
        self.connected = True
        return True
    
    def get_signal(self):
        """Generate trading signals from market data"""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 50)
        if rates is None:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Simple moving average crossover strategy
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        if last_row['sma20'] > last_row['sma50'] and prev_row['sma20'] <= prev_row['sma50']:
            return {'action': 'buy', 'price': last_row['close']}
        elif last_row['sma20'] < last_row['sma50'] and prev_row['sma20'] >= prev_row['sma50']:
            return {'action': 'sell', 'price': last_row['close']}
        return None

engine = TradingEngine()

@app.route('/signal', methods=['GET'])
def get_signal():
    if not engine.connected:
        return jsonify({'error': 'MT5 not connected'}), 500
    
    signal = engine.get_signal()
    if signal:
        return jsonify(signal)
    return jsonify({'action': 'hold'})

@app.route('/execute', methods=['POST'])
def execute_trade():
    data = request.json
    action = data.get('action')
    price = float(data.get('price', 0))
    lot_size = float(data.get('lot_size', 0.1))
    
    if action not in ['buy', 'sell']:
        return jsonify({'status': 'error', 'message': 'Invalid action'}), 400
    
    request_dict = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': SYMBOL,
        'volume': lot_size,
        'type': mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
        'price': price,
        'deviation': 10,
        'magic': 123456,
        'comment': 'REST API Trade',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request_dict)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({'status': 'error', 'code': result.retcode, 'message': result.comment}), 400
    
    return jsonify({
        'status': 'success',
        'order_id': result.order,
        'price': result.price
    })

def run_api():
    app.run(host='0.0.0.0', port=PORT, threaded=True)

if __name__ == '__main__':
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    print(f"REST API running on http://localhost:{PORT}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        mt5.shutdown()