import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
#Fetch data 
# (e.g., Gold Futures)
data = yf.download("GC=F", period="1d", interval="5m")
#flatten multi-level columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
#ensure column names are lowercase for consistency
data.columns = [col.lower() for col in data.columns]
#extract close prices
close_prices = data['close'].values.astype(float)
#confirm output
print(data)
#works well

#Determining S&R levels using the highest high and lowest low of the data
def detect_sr_levels(data, left_bars=15, right_bars=15):
    """
    Detect multiple support and resistance levels throughout the dataset.
    """
    # Calculate pivot highs
    data['pivot_high'] = data['high'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == max(x) else np.nan, raw=False)
    
    # Calculate pivot lows
    data['pivot_low'] = data['low'].rolling(window=left_bars+right_bars+1, center=True).apply(
        lambda x: x.iloc[left_bars] if x.iloc[left_bars] == min(x) else np.nan, raw=False)
    
    # Extract all pivot highs and lows
    pivot_highs = data['pivot_high'].dropna()
    pivot_lows = data['pivot_low'].dropna()
    
    return pivot_highs, pivot_lows

#Fetch the support and resistance levels
pivot_highs, pivot_lows = detect_sr_levels(data)
print("Pivot Highs:", pivot_highs)
print("Pivot Lows:", pivot_lows)

@app.route('/signal', methods=['GET'])
def function():
    return jsonify({'pivot_highs': pivot_highs.tolist(), 'pivot_lows': pivot_lows.tolist()})

if __name__ == '__main__':
    app.run(debug=False)