import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def flatten_multiindex(df):
    """Flatten MultiIndex columns to single level"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

def pivot_high_lows(data, left_bars=5, right_bars=2):
    """
    Identifies pivot highs and lows with robust column handling
    """
    # Flatten columns if MultiIndex exists
    data = flatten_multiindex(data)
    
    # Standardize column names (case insensitive)
    data.columns = data.columns.str.lower()
    
    try:
        highs = data['high_gc=f'].values if 'high_gc=f' in data.columns else data['high'].values
        lows = data['low_gc=f'].values if 'low_gc=f' in data.columns else data['low'].values
    except KeyError as e:
        available_cols = ", ".join(data.columns)
        raise KeyError(f"Required columns not found. Available columns: {available_cols}") from e
    
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    
    for i in range(left_bars, len(data) - right_bars):
        high_window = highs[i-left_bars:i+right_bars+1]
        low_window = lows[i-left_bars:i+right_bars+1]
        
        if highs[i] == np.max(high_window):
            pivot_highs[i] = highs[i]
            
        if lows[i] == np.min(low_window):
            pivot_lows[i] = lows[i]
    
    return pivot_highs, pivot_lows

def calculate_hh_hl_ll_lh(pivot_highs, pivot_lows, highs, lows):
    """Calculate HH/HL/LL/LH patterns with input validation"""
    hh = np.full(len(pivot_highs), False)
    hl = np.full(len(pivot_highs), False)
    ll = np.full(len(pivot_lows), False)
    lh = np.full(len(pivot_lows), False)
    
    ph_idx = np.where(~np.isnan(pivot_highs))[0]
    pl_idx = np.where(~np.isnan(pivot_lows))[0]
    
    for i in range(1, len(ph_idx)):
        current_high = pivot_highs[ph_idx[i]]
        prev_high = pivot_highs[ph_idx[i-1]]
        
        if current_high > prev_high:
            hh[ph_idx[i]] = True
        else:
            hl[ph_idx[i]] = True
    
    for i in range(1, len(pl_idx)):
        current_low = pivot_lows[pl_idx[i]]
        prev_low = pivot_lows[pl_idx[i-1]]
        
        if current_low < prev_low:
            ll[pl_idx[i]] = True
        else:
            lh[pl_idx[i]] = True
    
    return hh, hl, ll, lh

def plot_pivots(data, pivot_highs, pivot_lows, hh, hl, ll, lh):
    """Plot the results with error handling"""
    plt.figure(figsize=(15, 8))
    
    # Get price column (handle both flattened and regular formats)
    price_col = 'close_gc=f' if 'close_gc=f' in data.columns else 'close'
    
    # Plot price
    plt.plot(data.index, data[price_col], label='Price', color='black', alpha=0.7)
    
    # Plot patterns
    plot_kwargs = {
        's': 100,
        'linewidths': 1.5
    }
    
    high_col = 'high_gc=f' if 'high_gc=f' in data.columns else 'high'
    low_col = 'low_gc=f' if 'low_gc=f' in data.columns else 'low'
    
    plt.scatter(data.index, pivot_highs, color='red', marker='v', label='Pivot High')
    plt.scatter(data.index, pivot_lows, color='blue', marker='^', label='Pivot Low')
    
    plt.scatter(data.index[hh], data[high_col].values[hh], 
                color='green', marker='s', label='Higher High', **plot_kwargs)
    plt.scatter(data.index[hl], data[high_col].values[hl],
                color='green', marker='s', facecolors='none', 
                label='Higher Low', **plot_kwargs)
    plt.scatter(data.index[ll], data[low_col].values[ll],
                color='red', marker='s', label='Lower Low', **plot_kwargs)
    plt.scatter(data.index[lh], data[low_col].values[lh],
                color='red', marker='s', facecolors='none', 
                label='Lower High', **plot_kwargs)
    
    plt.title('Gold (GC=F) Highs & Lows Analysis')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    try:
        # Fetch data with proper parameters
        data = yf.download("EURUSD=X", period="1d", interval="1h", auto_adjust=False)
        
        # Debug: Print available columns
        print("Available columns:", data.columns.tolist())
        
        # Calculate indicators
        pivot_highs, pivot_lows = pivot_high_lows(data)
        
        # Get the correct column names for highs and lows
        high_col = 'high_gc=f' if 'high_gc=f' in data.columns else 'high'
        low_col = 'low_gc=f' if 'low_gc=f' in data.columns else 'low'
        
        hh, hl, ll, lh = calculate_hh_hl_ll_lh(
            pivot_highs, pivot_lows, 
            data[high_col].values, data[low_col].values
        )
        
        # Plot results
        plot_pivots(data, pivot_highs, pivot_lows, hh, hl, ll, lh)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Data sample:")
        print(data.head() if 'data' in locals() else "No data loaded")