import yfinance as yf
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def fetch_latest_btc_price():
    # Fetch the latest BTC price
    ticker = yf.Ticker("BTC-USD")
    latest_price = ticker.history(period="1d", interval="1m")
    if not latest_price.empty:
        latest_close = latest_price['Close'].iloc[-1]
        return latest_close
    else:
        return None

def fetch_columns(symbol: str, period: str, interval: str):
    # Fetch the latest BTC price
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

def detect_trend(row):
    if pd.isna(row['sma_5']) or pd.isna(row['sma_10']):
        return 'neutral'
    elif row['sma_5'] > row['sma_10']:
        return 'uptrend'
    elif row['sma_5'] < row['sma_10']:
        return 'downtrend'
    else:
        return 'neutral'

def calculate_sma():
    # Fetch the latest BTC price
    data = fetch_columns("BTC-USD", "1d", "1h")
    # Calculate 50-period and 200-period SMAs
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_10'] = data['close'].rolling(window=15).mean()
    # Detect trend
    data['trend'] = data.apply(detect_trend, axis=1)
    
    return data[['close', 'sma_5', 'sma_10', 'trend']]

def detectTrend(data):
    data = data.copy()
    data['trend'] = 'neutral'
    max_high = data['high'].max()
    min_high = data['high'].min()
    midpoint = (max_high + min_high) / 2
    data.loc[data['high'] > midpoint, 'trend'] = 'uptrend'
    data.loc[data['high'] < midpoint, 'trend'] = 'downtrend'

    return data

def calculate_fibonacci(data):
    """
    Calculates Fibonacci retracement levels based on the latest trend.
    Expects a 'trend' column and OHLC data.
    Returns a dictionary of retracement levels.
    """
    latest_trend = detectTrend(data)
    # Extract the trend value from the DataFrame
    trend = latest_trend['trend'].iloc[-1]  # Get the last row's trend value

    if trend == 'uptrend':
        swing_low = data['low'].min()
        swing_high = data['high'].max()
        levels = {
            '0.0%': swing_low,
            '23.6%': swing_high - 0.236 * (swing_high - swing_low),
            '38.2%': swing_high - 0.382 * (swing_high - swing_low),
            '50.0%': swing_high - 0.5 * (swing_high - swing_low),
            '61.8%': swing_high - 0.618 * (swing_high - swing_low),
            '78.6%': swing_high - 0.786 * (swing_high - swing_low),
            '100.0%': swing_high
        }
    elif trend == 'downtrend':
        swing_high = data['high'].max()
        swing_low = data['low'].min()
        levels = {
            '0.0%': swing_high,
            '23.6%': swing_low + 0.236 * (swing_high - swing_low),
            '38.2%': swing_low + 0.382 * (swing_high - swing_low),
            '50.0%': swing_low + 0.5 * (swing_high - swing_low),
            '61.8%': swing_low + 0.618 * (swing_high - swing_low),
            '78.6%': swing_low + 0.786 * (swing_high - swing_low),
            '100.0%': swing_low
        }
    else:
        levels = {"message": "Neutral trend, no retracement levels calculated."}

    return levels

def is_doji(row, body_threshold=0.05, shadow_ratio=2):
    """
    Identifies 4 types of Doji:
    1. Standard Doji - Small body with balanced wicks
    2. Long-Legged Doji - Small body with long upper/lower wicks
    3. Dragonfly Doji - No upper wick (bullish reversal)
    4. Gravestone Doji - No lower wick (bearish reversal)
    
    Parameters:
    - body_threshold: Max body size (5% of candle range default)
    - shadow_ratio: Wick-to-body ratio (2:1 default)
    """
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

if __name__ == "__main__":
    fetchData = fetch_columns("BTC-USD", "1d", "1h")
    print(fetchData)
    latest_price = fetch_latest_btc_price()
    fib_levels = calculate_fibonacci(fetchData)  # Pass the fetched data here
    trendData = detectTrend(fetchData)
    print("\nTrend detection result:")
    print(trendData[['close', 'high', 'low', 'trend']].tail())
    if latest_price is not None:
        print(f"Latest BTC Price: ${latest_price:.2f}")
    else:
        print("Failed to fetch the latest BTC price.")
    sma_trend_data = calculate_sma()
    print("\nSMA + Trend Data:")
    print(sma_trend_data.tail())
    print("Fibonacci Retracement Levels:")
    for level, price in fib_levels.items():
        print(f"{level}: {price:.2f}")