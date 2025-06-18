from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List, Tuple
from flask import jsonify  # Add this at the top with other imports

app = Flask(__name__)

# ----------------------------
# Technical Analysis Functions
# ----------------------------
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    
    print(f"Fetching data for {symbol}...")
    data = yf.download(tickers=symbol, period=period, interval=interval)
    # Flatten multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # Lowercase column names for consistency
    data.columns = [col.lower() for col in data.columns]

    # Sanity check for necessary columns
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(set(data.columns)):
        raise ValueError(f"Missing required OHLCV columns. Found: {data.columns}")

    return data

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    # SMAs
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_10'] = data['close'].rolling(window=10).mean()
    
    # Trend detection
    data['trend'] = data.apply(
        lambda row: 'uptrend' if row['sma_5'] > row['sma_10'] else 
                   'downtrend' if row['sma_5'] < row['sma_10'] else 'neutral',
        axis=1
    )
    
    # Volume oscillator
    short_ema = data['volume'].ewm(span=5, adjust=False).mean()
    long_ema = data['volume'].ewm(span=20, adjust=False).mean()
    data['volume_osc'] = 100 * (short_ema - long_ema) / long_ema
    
    return data

def detect_sr_breaks(data: pd.DataFrame) -> Dict[str, Any]:
    """Support/Resistance Breakout & Proximity Detection"""
    left_bars, right_bars, volume_thresh = 15, 15, 20
    current_price = data['close'].iloc[-1]
    
    data.loc[:, 'pivot_high'] = data['high'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == max(x) else np.nan, raw=False)

    data.loc[:, 'pivot_low'] = data['low'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == min(x) else np.nan, raw=False)

    last_resistance = data['pivot_high'].ffill().iloc[-1]
    last_support = data['pivot_low'].ffill().iloc[-1]

    resistance_break = (
        (data['close'] > last_resistance) & 
        (data['volume_osc'] > volume_thresh) &
        (~(data['open'] - data['low'] > data['close'] - data['open']))
    )

    support_break = (
        (data['close'] < last_support) &
        (data['volume_osc'] > volume_thresh) &
        (~(data['open'] - data['close'] < data['high'] - data['open']))
    )

    # Proximity check (~1% threshold)
    is_near_support = abs(current_price - last_support) / current_price < 0.01
    is_near_resistance = abs(current_price - last_resistance) / current_price < 0.01

    return {
        'support': last_support,
        'resistance': last_resistance,
        'bullish_break': bool(resistance_break.iloc[-1]),
        'bearish_break': bool(support_break.iloc[-1]),
        'is_near_support': is_near_support,
        'is_near_resistance': is_near_resistance
    }


def calculate_fibonacci(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate correct Fibonacci retracement levels based on trend"""
    swing_high = data['high'].max()
    swing_low = data['low'].min()
    trend = data['trend'].iloc[-1]

    diff = swing_high - swing_low

    if trend == 'uptrend':
        return {
            '100.0%': swing_low,
            '71.8%': swing_low + 0.786 * diff,
            '61.8%': swing_low + 0.618 * diff,
            '50.0%': swing_low + 0.5 * diff,
            '38.2%': swing_low + 0.382 * diff,
            '23.6%': swing_low + 0.236 * diff,
            '0.0%': swing_high
        }
    else:  # downtrend
        return {
            '100.0%': swing_high,
            '71.8%': swing_low + 0.786 * diff,
            '61.8%': swing_high - 0.618 * diff,
            '50.0%': swing_high - 0.5 * diff,
            '38.2%': swing_high - 0.382 * diff,
            '23.6%': swing_high - 0.236 * diff,
            '0.0%': swing_low
        }

                 
def is_doji(row, body_threshold=0.05, shadow_ratio=2):
    
    candle_range = row['high'] - row['low']
    body_size = abs(row['close'] - row['open'])
    
    # Avoid division by zero
    if candle_range == 0:
        return {'is_doji': False, 'type': None}
    
    body_percentage = body_size / candle_range
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']
    
    # Standard Doji condition
    is_standard = (body_percentage < body_threshold) and \
                  (upper_wick > shadow_ratio * body_size) and \
                  (lower_wick > shadow_ratio * body_size)
    
    # Long-Legged Doji (more extreme wicks)
    is_long_legged = (body_percentage < body_threshold) and \
                     (upper_wick > 3 * body_size) and \
                     (lower_wick > 3 * body_size)
    
    # Dragonfly Doji (bullish)
    is_dragonfly = (body_percentage < body_threshold) and \
                   (upper_wick < 0.1 * candle_range) and \
                   (lower_wick > shadow_ratio * body_size) and \
                   (row['close'] >= row['open'])  # Closing near high
    
    # Gravestone Doji (bearish)
    is_gravestone = (body_percentage < body_threshold) and \
                    (lower_wick < 0.1 * candle_range) and \
                    (upper_wick > shadow_ratio * body_size) and \
                    (row['close'] <= row['open'])  # Closing near low
    
    return {
        'is_doji': is_standard or is_long_legged or is_dragonfly or is_gravestone,
        'type': 'standard' if is_standard else \
                'long_legged' if is_long_legged else \
                'dragonfly' if is_dragonfly else \
                'gravestone' if is_gravestone else None
    }

def detect_candle_patterns(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect comprehensive candlestick patterns comparing n-2, n-1 and current candles
    Returns dictionary with all detected patterns
    """
    if len(data) < 3:
        return {
            'hammer': False,
            'inverted_hammer': False,
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'morning_star': False,
            'evening_star': False,
            'doji': False,
            'doji_type': None
        }
    
    # Get candles (current is incomplete, so we use n-1 as most recent complete)
    current = data.iloc[-1]   # n-1 (most recent complete candle)
    prev = data.iloc[-2]      # n-2
    prev_prev = data.iloc[-3] # n-3
    
    # 1. Hammer/Inverted Hammer (single candle patterns)
    body = abs(current['close'] - current['open'])
    lower_wick = min(current['open'], current['close']) - current['low']
    upper_wick = current['high'] - max(current['open'], current['close'])
    
    is_hammer = (lower_wick > 2 * body) and (upper_wick < 0.2 * body)
    is_inv_hammer = (upper_wick > 2 * body) and (lower_wick < 0.2 * body)
    
    # 2. Engulfing patterns (two-candle patterns)
    is_bull_engulf = (prev['close'] < prev['open'] and
                     current['close'] > current['open'] and
                     current['open'] < prev['close'] and
                     current['close'] > prev['open'])
    
    is_bear_engulf = (prev['close'] > prev['open'] and
                     current['close'] < current['open'] and
                     current['open'] > prev['close'] and
                     current['close'] < prev['open'])
    
    # 3. Morning/Evening Star (three-candle patterns) 
    is_morning_star = (prev_prev['close'] < prev_prev['open'] and
                      abs(prev_prev['close'] - prev_prev['open']) > (prev_prev['high'] - prev_prev['low']) * 0.7 and
                      prev['high'] < prev_prev['low'] and
                      current['close'] > current['open'] and
                      current['close'] > (prev_prev['open'] + prev_prev['close'])/2)
    
    is_evening_star = (prev_prev['close'] > prev_prev['open'] and
                      abs(prev_prev['close'] - prev_prev['open']) > (prev_prev['high'] - prev_prev['low']) * 0.7 and
                      prev['low'] > prev_prev['high'] and
                      current['close'] < current['open'] and
                      current['close'] < (prev_prev['open'] + prev_prev['close'])/2)
    
    # 4. Doji patterns
    doji_info = is_doji(current)
    
    return {
        'hammer': is_hammer,
        'inverted_hammer': is_inv_hammer,
        'bullish_engulfing': is_bull_engulf,
        'bearish_engulfing': is_bear_engulf,
        'morning_star': is_morning_star,
        'evening_star': is_evening_star,
        'doji': doji_info['is_doji'],
        'doji_type': doji_info['type']
    }
      
def add_fibonacci_levels(fig: go.Figure, data: pd.DataFrame) -> go.Figure:
   
    # Calculate Fibonacci levels (using your existing function)
    fib_levels = calculate_fibonacci(data)
    
    # Add each Fibonacci level as a horizontal line
    for level, price in fib_levels.items():
        if isinstance(price, (int, float)):  # Skip non-numeric levels
            fig.add_hline(
                y=price,
                line_dash="dot",
                line_color="gray",
                opacity=0.7,
                annotation_text=f"Fib {level}",
                annotation_position="right"
            )
    return fig

def generate_trading_signal(data: pd.DataFrame, current_price: float, last_signal: str = "HOLD") -> Dict[str, Any]:
    if len(data) < 20:
        return {"decision": "HOLD", "confidence": 0, "details": {}}
    
    indicators = {
        "patterns": detect_candle_patterns(data),
        "fib": calculate_fibonacci(data),
        "sma_trend": data['trend'].iloc[-1],
        "sr": detect_sr_breaks(data),
        "volume_osc": data['volume_osc'].iloc[-1]
    }

    score = 0
    reasons = []
    
    # -- Candlestick Scoring
    bullish_score = 0
    bearish_score = 0
    
    if indicators["patterns"]["morning_star"]:
        bullish_score += 3
        reasons.append("Morning Star")
    if indicators["patterns"]["bullish_engulfing"]:
        bullish_score += 2
        reasons.append("Bullish Engulfing")
    if indicators["patterns"]["hammer"]:
        bullish_score += 1.5
        reasons.append("Hammer")

    if indicators["patterns"]["evening_star"]:
        bearish_score += 3
        reasons.append("Evening Star")
    if indicators["patterns"]["bearish_engulfing"]:
        bearish_score += 2
        reasons.append("Bearish Engulfing")
    if indicators["patterns"]["inverted_hammer"]:
        bearish_score += 1.5
        reasons.append("Inverted Hammer")

    score += bullish_score - bearish_score

    # -- Fibonacci Level Influence
    fib_distances = {k: abs(current_price - v) for k, v in indicators["fib"].items()
                     if isinstance(v, (float, int))}
    nearest_fib = min(fib_distances.items(), key=lambda x: x[1]) if fib_distances else (None, None)
    fib_level, fib_distance = nearest_fib
    
   # Confluence of trend + Fibonacci level
    if fib_level in ["38.2%", "50.0%", "61.8%"] and fib_distance < current_price * 0.01:
        if indicators["sma_trend"] == "uptrend":
           score += 1.5
           reasons.append(f"Bounce Near Fib Support ({fib_level})")
        elif indicators["sma_trend"] == "downtrend":
             score -= 1.5
             reasons.append(f"Rejected Near Fib Resistance ({fib_level})")

# Extra scoring for price bouncing off support/resistance in the trend direction
    if indicators["sma_trend"] == "uptrend" and indicators["sr"].get("is_near_support", False):
             score += 1.5
             reasons.append("Uptrend + Near Support (Confluence)")

    if indicators["sma_trend"] == "downtrend" and indicators["sr"].get("is_near_resistance", False):
         score -= 1.5
         reasons.append("Downtrend + Near Resistance (Confluence)")

    # -- Trend
    if indicators["sma_trend"] == "uptrend":
        score += 1
    elif indicators["sma_trend"] == "downtrend":
        score -= 1

    # -- Support / Resistance with Volume
    if indicators["sr"]["bullish_break"] and indicators["volume_osc"] > 20:
        score += 2
        reasons.append("Bullish SR Breakout w/ Volume")
    if indicators["sr"]["bearish_break"] and indicators["volume_osc"] > 20:
        score -= 2
        reasons.append("Bearish SR Breakdown w/ Volume")

    # -- Final Decision
    confidence = round(min(100, abs(score) * 10), 2)
    signal = "HOLD"
    
    data['sma_1'] = data['close'].rolling(window=2).mean()
    data['sma_10'] = data['close'].rolling(window=15).mean()
    
    if score >= 3 and  confidence >= 40 and data['sma_1'].iloc[-1] > data['sma_10'].iloc[-1]:
        signal = "BUY"
    elif score <= -3 and confidence >= 40 and data['sma_1'].iloc[-1] < data['sma_10'].iloc[-1]:
        signal = "SELL"
    track = 0
    if(data['sma_1'].iloc[-1] > data['sma_10'].iloc[-1]):
           track = 1
    elif(data['sma_1'].iloc[-1] < data['sma_10'].iloc[-1]):
            track = -1  

    return {
        "decision": signal,
        "confidence": confidence,
        "reasons": reasons,
        "track": track,
        # "details": {
          #   "patterns": ["No strong signals"],
           #     "fib_level": nearest_fib[0],
           #     "trend": indicators["sma_trend"],
           #     "sr_status": indicators["sr"]
      #  }
    }

def backtest_strategy(
    symbol: str = "GC=F",
    period: str = "1y",
    interval: str = "1h",
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.02,  # 2% risk per trade
    commission: float = 0.0005,  # 0.05% commission
) -> Dict[str, any]:
    """
    Backtests the trading strategy on historical data
    Returns performance metrics and trade-by-trade results
    """
    print("Starting backtest...")
    print("fetching data...")
    print("period:", period)
    print("interval:", interval)
    # 1. Fetch and prepare data
    data = fetch_data(symbol, period, interval)
    data = calculate_indicators(data)
    
    # 2. Initialize tracking variables
    balance = initial_balance
    equity = [balance]
    trades = []
    position = None
    trade_id = 0
    
    # 3. Main backtesting loop
    for i in range(2, len(data)):
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Generate signal (using your existing function)
        signal = generate_trading_signal(data[:i+1], current['close'])
        
        # Close existing position if needed
        if position:
            if (position['type'] == 'LONG' and (
                current['low'] <= position['sl'] or 
                current['high'] >= position['tp'] or
                signal['decision'] == 'SELL'
            )):
                exit_price = (
                    position['tp'] if current['high'] >= position['tp'] else
                    position['sl'] if current['low'] <= position['sl'] else
                    current['close']
                )
                pnl = (exit_price - position['entry']) * position['size'] - commission
                balance += pnl
                trades.append({
                    'id': trade_id,
                    'type': 'LONG',
                    'entry': position['entry'],
                    'exit': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'entry_time': position['time'],
                    'exit_time': current.name,
                    'signal': position['signal']
                })
                position = None
                
            elif (position['type'] == 'SHORT' and (
                current['high'] >= position['sl'] or
                current['low'] <= position['tp'] or
                signal['decision'] == 'BUY'
            )):
                exit_price = (
                    position['tp'] if current['low'] <= position['tp'] else
                    position['sl'] if current['high'] >= position['sl'] else
                    current['close']
                )
                pnl = (position['entry'] - exit_price) * position['size'] - commission
                balance += pnl
                trades.append({
                    'id': trade_id,
                    'type': 'SHORT',
                    'entry': position['entry'],
                    'exit': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'entry_time': position['time'],
                    'exit_time': current.name,
                    'signal': position['signal']
                })
                position = None
                trade_id += 1
        
        # Open new position if no current position and strong signal
        if not position and signal['decision'] in ('BUY', 'SELL') and signal['confidence'] >= 50:
            risk_amount = balance * risk_per_trade
            if signal['decision'] == 'BUY':
                entry_price = current['close']
                sl = entry_price * (1 - 0.01)  # 1% stop loss (adjust as needed)
                tp = entry_price * (1 + 0.02)  # 2% take profit
                size = risk_amount / (entry_price - sl)
                position = {
                    'type': 'LONG',
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'time': current.name,
                    'signal': signal
                }
            else:  # SELL
                entry_price = current['close']
                sl = entry_price * (1 + 0.01)  # 1% stop loss
                tp = entry_price * (1 - 0.02)  # 2% take profit
                size = risk_amount / (sl - entry_price)
                position = {
                    'type': 'SHORT',
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'size': size,
                    'time': current.name,
                    'signal': signal
                }
        
        equity.append(balance)
    
    # 4. Calculate performance metrics
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum() + initial_balance
        trades_df['returns'] = trades_df['pnl'] / initial_balance
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        metrics = {
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance * 100,
            'total_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_win': wins['pnl'].mean() if not wins.empty else 0,
            'avg_loss': losses['pnl'].mean() if not losses.empty else 0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if not losses.empty else float('inf'),
            'max_drawdown': (min(equity) - initial_balance) / initial_balance * 100,
            'sharpe_ratio': calculate_sharpe_ratio(trades_df['returns']) if len(trades_df) > 1 else 0,
            'trades': trades_df.to_dict('records')
        }
    else:
        metrics = {
            'final_balance': initial_balance,
            'total_return': 0,
            'total_trades': 0,
            'message': "No trades executed during this period"
        }
    
    return metrics

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates annualized Sharpe ratio from trade returns"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

def log_results_to_file(results: Dict[str, Any], timeframe: str = "1H"):
    """Logs backtest results to a file."""
    log_file = "results_log.txt"
    with open(log_file, "a", encoding="utf-8") as file:  # Specify utf-8 encoding
        file.write("=====================EURUSD=============================\n")
        file.write(f"🕒 Timeframe Tested: {timeframe}\n")
        file.write(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Final Balance: ${results['final_balance']:.2f}\n")
        file.write(f"Total Return: {results['total_return']:.2f}%\n")
        file.write(f"Total Trades: {results['total_trades']}\n")
        file.write(f"Win Rate: {results['win_rate']:.2f}%\n")
        file.write(f"Average Win: ${results['avg_win']:.2f}\n")
        file.write(f"Average Loss: ${results['avg_loss']:.2f}\n")
        file.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
        file.write(f"Max Drawdown: {results['max_drawdown']:.2f}%\n")
        file.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
        file.write("--------------------------------------------------\n")
        for trade in results['trades']:
            file.write(f"Trade ID: {trade['id']}, Type: {trade['type']}, "
                       f"Entry: {trade['entry']:.2f}, Exit: {trade['exit']:.2f}, "
                       f"PnL: {trade['pnl']:.2f}, Entry Time: {trade['entry_time']}, "
                       f"Exit Time: {trade['exit_time']}\n")
        file.write("==================================================\n\n")

# ----------------------------
# Flask Application
# ----------------------------
@app.route('/')
def dashboard():
    # Fetch and process data
    data = fetch_data("GC=F", "2d", "1h")
    data = calculate_indicators(data)
    
    # Technical signals
    sr_signals = detect_sr_breaks(data)
    fib_levels = calculate_fibonacci(data)
    candle_patterns = detect_candle_patterns(data)
    latest_price = data['close'].iloc[-1]
    
    
# Generate signal
    signal = generate_trading_signal(data, latest_price)
    print(f"Decision: {signal['decision']} ({signal['confidence']}% confidence)")
  
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='BTC/USD'
        ),
        go.Scatter(
            x=data.index,
            y=data['sma_5'],
            line=dict(color='orange', width=1),
            name='SMA 5'
        ),
        go.Scatter(
            x=data.index,
            y=data['sma_10'],
            line=dict(color='purple', width=1),
            name='SMA 10'
        )
    ])
    
    # Add support/resistance lines
    fig.add_hline(y=sr_signals['support'], line_dash="dot", 
                 annotation_text="Support", line_color="blue")
    fig.add_hline(y=sr_signals['resistance'], line_dash="dot", 
                 annotation_text="Resistance", line_color="red")
    
    # Update layout
    fig.update_layout(
        title='BTC/USD Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    fig = add_fibonacci_levels(fig, data)
    # Prepare context for template
    context = {
        'chart': fig.to_html(full_html=False),
        'price': f"{latest_price:.2f}",
        'trend': data['trend'].iloc[-1],
        'sr_signals': sr_signals,
        'fib_levels': fib_levels,
        'candle_patterns': candle_patterns,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('dashboard.html', **context)

@app.route('/api/signal')
def api_signal():
    """Dedicated JSON endpoint for trading signals"""
    try:
        # Fetch and process data
        data = fetch_data("GC=F", "2d", "1h")
        data = calculate_indicators(data)
        
        # Generate signal
        signal = generate_trading_signal(data, data['close'].iloc[-1])
        
        # Add timestamp
        signal['timestamp'] = datetime.now().isoformat()
        
        return jsonify(signal)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "decision": "HOLD",
            "confidence": 0
        }), 500  
if __name__ == '__main__':
    results = backtest_strategy(period="2d", interval="5m")
    print(f"Strategy Results ({len(results['trades'])} trades):")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    log_results_to_file(results, timeframe="5m")  

    app.run(debug=False)
   
