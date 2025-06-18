# app.py
#
# --- Project Structure ---
# .
# ├── app.py              <-- This file
# ├── modules/
# │   ├── __init__.py     <-- An empty file
# │   ├── db.py           <-- Your database module
# │   └── analysis.py     <-- Your advanced analysis logic
# ├── .env
# └── your_service_account_key.json
#
# --- To Run ---
# 1. pip install Flask yfinance pandas numpy plotly firebase-admin psycopg2-binary python-dotenv
# 2. Set up your .env file and Firebase key.
# 3. Run: python app.py

from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps

# --- Local Module Imports ---
from modules import db
from modules import analysis # Using your advanced analysis module

# --- Initialization ---
app = Flask(__name__)

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("path/to/your/serviceAccountKey.json") 
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")

# Create database tables
try:
    db.create_tables()
    print("Database tables checked/created successfully.")
except Exception as e:
    print(f"Error connecting to or creating database tables: {e}")

# --- Authentication Decorator ---
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Authorization header is missing or invalid'}), 401
        
        id_token = auth_header.split('Bearer ')[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            kwargs['uid'] = decoded_token['uid']
            kwargs['email'] = decoded_token.get('email')
        except Exception as e:
            return jsonify({'message': f'Error verifying token: {e}'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

# --- API Endpoints ---
@app.route('/api/register', methods=['POST'])
@token_required
def register_user_endpoint(uid, email):
    try:
        db.register_user(firebase_uid=uid, email=email)
        return jsonify({'status': 'success', 'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Database error: {e}'}), 500

@app.route('/api/signals', methods=['GET'])
@token_required
def get_signals_endpoint(uid, **kwargs):
    try:
        signals = db.get_signal_history(firebase_uid=uid)
        return jsonify(signals)
    except Exception as e:
        return jsonify({"error": f"Could not retrieve signal history: {e}"}), 500

@app.route('/api/analyze', methods=['POST'])
@token_required
def analyze_pairs(uid, **kwargs):
    """
    This endpoint now uses the advanced analysis module.
    It no longer calls an LLM, as the logic is self-contained.
    """
    data = request.get_json()
    if not data or 'pairs' not in data: return jsonify({"error": "Invalid request."}), 400
    
    pairs_to_analyze = data['pairs']
    full_analysis = {}

    for pair in pairs_to_analyze:
        try:
            print(f"--- Analyzing {pair} for user {uid} ---")
            
            # 1. Fetch data
            ticker_symbol = f"{pair}=X" if len(pair) == 6 else pair
            market_data = analysis.fetch_data(ticker_symbol, period="10d", interval="1h")
            if market_data.empty:
                print(f"No data for {pair}")
                continue

            # 2. Calculate all indicators using your advanced functions
            market_data = analysis.calculate_indicators(market_data)
            
            # 3. Generate the trading signal using your scoring logic
            current_price = market_data['close'].iloc[-1]
            signal_result = analysis.generate_trading_signal(market_data, current_price)
            
            # 4. Generate the Plotly chart HTML
            chart_html = analysis.generate_interactive_chart(market_data, pair)

            # 5. Structure the response
            full_analysis[pair] = {
                "ticker": pair,
                "current_price": f"{current_price:.5f}",
                "decision": signal_result['decision'],
                "confidence": signal_result['confidence'],
                "reasons": signal_result.get('reasons', []),
                "chart_html": chart_html, # Send the interactive chart to the frontend
                "details": signal_result.get('details', {})
            }

            # 6. Log the signal to the database
            try:
                db.log_signal(
                    firebase_uid=uid,
                    asset=pair,
                    recommendation=signal_result['decision'],
                    confidence_score=signal_result['confidence'],
                    rationale=", ".join(signal_result.get('reasons', [])),
                    price=current_price
                )
            except Exception as e:
                print(f"DB LOGGING FAILED for {pair}: {e}")

        except Exception as e:
            print(f"CRITICAL ERROR processing {pair}: {e}")
            full_analysis[pair] = {"error": str(e)}

    return jsonify(full_analysis)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# modules/analysis.py
# This module contains the advanced technical analysis, signal generation,
# and charting logic adapted from your scripts.

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

# ----------------------------
# Data Fetching
# ----------------------------
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches and prepares market data."""
    print(f"Fetching data for {symbol}...")
    data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
    if data.empty:
        return data
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data.columns = [col.lower() for col in data.columns]

    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(set(data.columns)):
        raise ValueError(f"Missing required OHLCV columns for {symbol}. Found: {data.columns}")

    return data

# ----------------------------
# Indicator Calculation
# ----------------------------
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all necessary technical indicators."""
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['trend'] = np.where(data['sma_5'] > data['sma_10'], 'uptrend', 'downtrend')
    
    short_ema = data['volume'].ewm(span=5, adjust=False).mean()
    long_ema = data['volume'].ewm(span=20, adjust=False).mean()
    data['volume_osc'] = 100 * (short_ema - long_ema) / long_ema
    
    return data

# ----------------------------
# Pattern & Level Detection
# ----------------------------
def detect_sr_breaks(data: pd.DataFrame) -> Dict[str, Any]:
    """Support/Resistance Breakout & Proximity Detection."""
    left_bars, right_bars, volume_thresh = 15, 15, 20
    current_price = data['close'].iloc[-1]
    
    data['pivot_high'] = data['high'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == max(x) else np.nan, raw=False).ffill()

    data['pivot_low'] = data['low'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == min(x) else np.nan, raw=False).ffill()

    last_resistance = data['pivot_high'].iloc[-1]
    last_support = data['pivot_low'].iloc[-1]

    resistance_break = current_price > last_resistance and data['volume_osc'].iloc[-1] > volume_thresh
    support_break = current_price < last_support and data['volume_osc'].iloc[-1] > volume_thresh

    is_near_support = abs(current_price - last_support) / current_price < 0.01
    is_near_resistance = abs(current_price - last_resistance) / current_price < 0.01

    return {
        'support': last_support, 'resistance': last_resistance,
        'bullish_break': bool(resistance_break), 'bearish_break': bool(support_break),
        'is_near_support': is_near_support, 'is_near_resistance': is_near_resistance
    }

def calculate_fibonacci(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    swing_high = data['high'].max()
    swing_low = data['low'].min()
    diff = swing_high - swing_low
    
    return {
        'level_100': swing_high, 'level_78_6': swing_high - 0.236 * diff,
        'level_61_8': swing_high - 0.382 * diff, 'level_50_0': swing_high - 0.5 * diff,
        'level_38_2': swing_high - 0.618 * diff, 'level_23_6': swing_high - 0.786 * diff,
        'level_0': swing_low
    }

def detect_candle_patterns(data: pd.DataFrame) -> Dict[str, bool]:
    """Detect key candlestick patterns from the last few candles."""
    if len(data) < 3: return {}
    
    current, prev, prev_prev = data.iloc[-1], data.iloc[-2], data.iloc[-3]
    
    body = abs(current['close'] - current['open'])
    lower_wick = min(current['open'], current['close']) - current['low']
    upper_wick = current['high'] - max(current['open'], current['close'])

    is_hammer = (lower_wick > 2 * body) and (upper_wick < 0.2 * body)
    is_bull_engulf = (prev['close'] < prev['open']) and (current['close'] > current['open']) and (current['close'] > prev['open']) and (current['open'] < prev['close'])
    is_bear_engulf = (prev['close'] > prev['open']) and (current['close'] < current['open']) and (current['close'] < prev['open']) and (current['open'] > prev['close'])
    is_morning_star = (prev_prev['close'] < prev_prev['open']) and (prev['high'] < prev_prev['low']) and (current['close'] > current['open']) and (current['close'] > (prev_prev['open'] + prev_prev['close'])/2)
    is_evening_star = (prev_prev['close'] > prev_prev['open']) and (prev['low'] > prev_prev['high']) and (current['close'] < current['open']) and (current['close'] < (prev_prev['open'] + prev_prev['close'])/2)

    return {
        'hammer': is_hammer, 'bullish_engulfing': is_bull_engulf, 'bearish_engulfing': is_bear_engulf,
        'morning_star': is_morning_star, 'evening_star': is_evening_star
    }

# ----------------------------
# Signal Generation
# ----------------------------
def generate_trading_signal(data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """Generate a trading signal based on a scoring system."""
    if len(data) < 20:
        return {"decision": "HOLD", "confidence": 0, "details": {}}
    
    patterns = detect_candle_patterns(data)
    sr = detect_sr_breaks(data)
    trend = data['trend'].iloc[-1]
    
    score = 0
    reasons = []

    # Pattern Scoring
    if patterns.get('morning_star'): score += 3; reasons.append("Morning Star")
    if patterns.get('bullish_engulfing'): score += 2; reasons.append("Bullish Engulfing")
    if patterns.get('hammer'): score += 1.5; reasons.append("Hammer")
    if patterns.get('evening_star'): score -= 3; reasons.append("Evening Star")
    if patterns.get('bearish_engulfing'): score -= 2; reasons.append("Bearish Engulfing")

    # Trend & S/R Confluence
    if trend == "uptrend":
        score += 1
        if sr.get("is_near_support"): score += 1.5; reasons.append("Uptrend near Support")
        if sr.get("bullish_break"): score += 2; reasons.append("Bullish Breakout")
    elif trend == "downtrend":
        score -= 1
        if sr.get("is_near_resistance"): score -= 1.5; reasons.append("Downtrend near Resistance")
        if sr.get("bearish_break"): score -= 2; reasons.append("Bearish Breakdown")

    confidence = min(100, abs(score) * 15)
    
    decision = "HOLD"
    if score >= 2.5 and confidence >= 40:
        decision = "BUY"
    elif score <= -2.5 and confidence >= 40:
        decision = "SELL"
        
    return { "decision": decision, "confidence": confidence, "reasons": reasons, "details": { "patterns": patterns, "sr": sr, "trend": trend } }

# ----------------------------
# Charting
# ----------------------------
def generate_interactive_chart(data: pd.DataFrame, title: str) -> str:
    """Generates an interactive Plotly chart and returns it as HTML."""
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name=title, increasing_line_color='green', decreasing_line_color='red'
    ))

    # SMAs
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_5'], line=dict(color='orange', width=1), name='SMA 5'))
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_10'], line=dict(color='purple', width=1), name='SMA 10'))

    # S/R Levels
    sr = detect_sr_breaks(data)
    fig.add_hline(y=sr['support'], line_dash="dot", annotation_text="Support", line_color="cyan")
    fig.add_hline(y=sr['resistance'], line_dash="dot", annotation_text="Resistance", line_color="magenta")
    
    # Fibonacci Levels
    fib_levels = calculate_fibonacci(data)
    for level, price in fib_levels.items():
        fig.add_hline(y=price, line_dash="dash", line_color="gray", opacity=0.5,
                      annotation_text=level.replace('_', ' ').replace('level', 'Fib'), annotation_position="right")

    fig.update_layout(
        title=f'{title} Technical Analysis',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
