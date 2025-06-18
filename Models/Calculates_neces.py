import yfinance as yf
import pandas as pd
import yfinance as yf
import numpy as np
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
    
#Fecth the list of all prices in a range
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
    data = fetch_columns("BTC-USD", "7d", "1h")
    
    # Calculate 50-period and 200-period SMAs
    data['sma_5'] = data['close'].rolling(window=5).mean()
    data['sma_10'] = data['close'].rolling(window=15).mean()
    
    # Detect trend
    data['trend'] = data.apply(detect_trend, axis=1)
    
    return data[['close', 'sma_5', 'sma_10', 'trend']]
#Determine trend using price data highest high and Lowest low
def detectTrend(data):
    data = data.copy()
    data['trend'] = 'neutral'

    max_high = data['high'].max()
    min_high = data['high'].min()
    midpoint = (max_high + min_high) / 2

    data.loc[data['high'] > midpoint, 'trend'] = 'uptrend'
    data.loc[data['high'] < midpoint, 'trend'] = 'downtrend'

    return data
# calculate fibbonaci retracement levels
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

#volume helper
def volume_oscillator(volume, short_window=5, long_window=10):
    """
    Exactly replicates LuxAlgo's volume % oscillator:
    osc = 100 * (ema(volume,5) - ema(volume,10)) / ema(volume,10)
    """
    short_ema = volume.ewm(span=short_window, adjust=False).mean()
    long_ema = volume.ewm(span=long_window, adjust=False).mean()
    return 100 * (short_ema - long_ema) / long_ema

#Determining S&R levels using the highest high and lowest low of the data
def detect_sr_breaks(data, left_bars=15, right_bars=15, volume_thresh=20):
    # Calculate pivot points (unchanged)
    # Calculate pivot highs/lows
    data['pivot_high'] = data['high'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == max(x) else np.nan, raw=False)
    
    data['pivot_low'] = data['low'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == min(x) else np.nan, raw=False)
    # PROPER VOLUME OSCILLATOR
    data['volume_osc'] = volume_oscillator(data['volume'])
    
    # Breakout conditions with volume filter
    resistance_break = (
        (data['close'] > data['pivot_high'].ffill()) & 
        (data['volume_osc'] > volume_thresh) &
        # Wick condition: Not a bearish wick-dominated candle
        (~(data['open'] - data['low'] > data['close'] - data['open']))
    )
    
    support_break = (
        (data['close'] < data['pivot_low'].ffill()) &
        (data['volume_osc'] > volume_thresh) &
        # Wick condition: Not a bullish wick-dominated candle
        (~(data['open'] - data['close'] < data['high'] - data['open']))
    )
    
    return {
        'support': data['pivot_low'].ffill().iloc[-1],
        'resistance': data['pivot_high'].ffill().iloc[-1],
        'bullish_break': resistance_break.iloc[-1],
        'bearish_break': support_break.iloc[-1],
        'volume_osc': data['volume_osc'].iloc[-1]  # For debugging
    }
    
def sr_breakout_score(data):
    sr = detect_sr_breaks(data)
    return {
        'S/R Breakout': {
            'Buy': 1 if sr['bullish_break'] else 0,
            'Sell': 1 if sr['bearish_break'] else 0
        }
    }    
    
    #determining Chart patterns
def is_hammer(row):
    body = abs(row['close'] - row['open'])
    lower_wick = row['open'] - row['low'] if row['close'] > row['open'] else row['close'] - row['low']
    upper_wick = row['high'] - row['close'] if row['close'] > row['open'] else row['high'] - row['open']
    
    return (lower_wick > 2 * body) and (upper_wick < 0.2 * body) and (row['close'] > row['open'])

def is_inverted_hammer(row):
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - row['close'] if row['close'] > row['open'] else row['high'] - row['open']
    lower_wick = row['open'] - row['low'] if row['close'] > row['open'] else row['close'] - row['low']
    
    return (upper_wick > 2 * body) and (lower_wick < 0.2 * body) and (row['close'] > row['open'])
#determing engulfing patterns
def is_bullish_engulfing(df, idx):
    current = df.iloc[idx]
    prev = df.iloc[idx-1]
    return (prev['close'] < prev['open']) and \
           (current['close'] > current['open']) and \
           (current['open'] < prev['close']) and \
           (current['close'] > prev['open'])

def is_bearish_engulfing(df, idx):
    current = df.iloc[idx]
    prev = df.iloc[idx-1]
    return (prev['close'] > prev['open']) and \
           (current['close'] < current['open']) and \
           (current['open'] > prev['close']) and \
           (current['close'] < prev['open'])
#determinign 
def is_morning_star(df, idx):
    day1 = df.iloc[idx-2]
    day2 = df.iloc[idx-1]
    day3 = df.iloc[idx]
    
    return (day1['close'] < day1['open']) and \
           (abs(day1['close'] - day1['open']) > day1['high'] - day1['low'] * 0.7) and \
           (day2['high'] < day1['low']) and \
           (day3['close'] > day3['open']) and \
           (day3['close'] > (day1['open'] + day1['close'])/2)

def is_evening_star(df, idx):
    day1 = df.iloc[idx-2]
    day2 = df.iloc[idx-1]
    day3 = df.iloc[idx]
    
    return (day1['close'] > day1['open']) and \
           (abs(day1['close'] - day1['open']) > day1['high'] - day1['low'] * 0.7) and \
           (day2['low'] > day1['high']) and \
           (day3['close'] < day3['open']) and \
           (day3['close'] < (day1['open'] + day1['close'])/2)
 
 
def detect_candle_patterns(data):
    patterns = {
        'hammer': [],
        'inverted_hammer': [],
        'bullish_engulfing': [],
        'bearish_engulfing': [],
        'morning_star': [],
        'evening_star': []
    }
    
    for i in range(2, len(data)):
        if is_hammer(data.iloc[i]):
            patterns['hammer'].append(i)
        if is_inverted_hammer(data.iloc[i]):
            patterns['inverted_hammer'].append(i)
        if i > 0 and is_bullish_engulfing(data, i):
            patterns['bullish_engulfing'].append(i)
        if i > 0 and is_bearish_engulfing(data, i):
            patterns['bearish_engulfing'].append(i)
        if i > 1 and is_morning_star(data, i):
            patterns['morning_star'].append(i)
        if i > 1 and is_evening_star(data, i):
            patterns['evening_star'].append(i)
    
    return patterns

def candle_pattern_scores(data):
    patterns = detect_candle_patterns(data)
    latest_idx = len(data) - 1
    
    return {
        'Candle Pattern': {
            'Buy': 1 if (latest_idx in patterns['hammer'] or 
                         latest_idx in patterns['inverted_hammer'] or 
                         latest_idx in patterns['bullish_engulfing'] or 
                         latest_idx in patterns['morning_star']) else 0,
            'Sell': 1 if (latest_idx in patterns['bearish_engulfing'] or 
                          latest_idx in patterns['evening_star']) else 0
        }
    }
def plot_patterns(data, patterns):
    plt.figure(figsize=(20,10))
    plt.plot(data['close'], label='Price', alpha=0.5)
    
    colors = {
        'hammer': 'green',
        'inverted_hammer': 'lime',
        'bullish_engulfing': 'blue',
        'bearish_engulfing': 'red',
        'morning_star': 'cyan',
        'evening_star': 'magenta'
    }
    
    for pattern, idxs in patterns.items():
        if idxs:
            plt.scatter(data.iloc[idxs].index, 
                       data.iloc[idxs]['close'], 
                       color=colors[pattern], 
                       label=pattern,
                       s=100)
    
    plt.legend()
                  
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

def detect_doji_with_context(data, trend_col='trend'):
    """
    Adds trend awareness for higher-probability signals:
    - Dragonfly in downtrend = Strong buy signal
    - Gravestone in uptrend = Strong sell signal
    """
    doji_signals = []
    for i in range(len(data)):
        row = data.iloc[i]
        doji_info = is_doji(row)
        
        if not doji_info['is_doji']:
            continue
            
        # Add trend context
        signal_strength = 0
        if doji_info['type'] == 'dragonfly' and data.iloc[i][trend_col] == 'downtrend':
            signal_strength = 2  # Strong bullish reversal
        elif doji_info['type'] == 'gravestone' and data.iloc[i][trend_col] == 'uptrend':
            signal_strength = 2  # Strong bearish reversal
        else:
            signal_strength = 1  # Neutral doji
            
        doji_signals.append({
            'index': i,
            'type': doji_info['type'],
            'strength': signal_strength,
            'price': row['close']
        })
    
    return doji_signals

def doji_scoring(data):
    signals = detect_doji_with_context(data)
    latest_signal = next((s for s in reversed(signals) if s['index'] == len(data)-1), None)
    
    return {
        'Doji Pattern': {
            'Buy': latest_signal['strength'] if latest_signal and latest_signal['type'] in ['dragonfly', 'long_legged'] else 0,
            'Sell': latest_signal['strength'] if latest_signal and latest_signal['type'] in ['gravestone', 'long_legged'] else 0
        }
    }
    
def plot_doji(data, signals):
    plt.figure(figsize=(20,10))
    plt.plot(data['close'], label='Price', alpha=0.7)
    
    colors = {
        'standard': 'gray',
        'long_legged': 'purple',
        'dragonfly': 'green',
        'gravestone': 'red'
    }
    
    for signal in signals:
        plt.scatter(data.index[signal['index']], 
                   signal['price'],
                   color=colors[signal['type']],
                   s=100 + 50*signal['strength'],  # Size indicates strength
                   label=f"{signal['type']} ({'Strong' if signal['strength']>1 else 'Neutral'})")
    
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    
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
        
    # Test
ffetchData = fetch_columns("BTC-USD", "1d", "1m")
sr_signals = detect_sr_breaks(ffetchData)

print(f"""
Current Volume Osc: {sr_signals['volume_osc']:.2f}%
Resistance Level: {sr_signals['resistance']:.2f}
Support Level: {sr_signals['support']:.2f}
Bullish Break: {sr_signals['bullish_break']}
Bearish Break: {sr_signals['bearish_break']}
""")

# Fetch data with trend information
data = fetch_columns("BTC-USD", "1d", "1m")


# Calculate SMA columns before detecting trends
data['sma_5'] = data['close'].rolling(window=5).mean()
data['sma_10'] = data['close'].rolling(window=10).mean()

# Detect trend using the SMA columns
data['trend'] = data.apply(detect_trend, axis=1)

# Detect and score
doji_signals = detect_doji_with_context(data)
scores = doji_scoring(data)

print(f"Latest Doji Signal: {scores}")
plot_doji(data, doji_signals)

# Detect patterns
patterns = detect_candle_patterns(data)
scores = candle_pattern_scores(data)

print(f"Latest Candle Scores: {scores}")
plot_patterns(data, patterns)