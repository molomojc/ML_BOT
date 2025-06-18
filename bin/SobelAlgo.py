import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
import pandas as pd

# Fetch data (e.g., Gold Futures)
data = yf.download("GC=F", period="1d", interval="15m")

# Flatten multi-level columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Ensure column names are lowercase for consistency
data.columns = [col.lower() for col in data.columns]

# Extract close prices
close_prices = data['close'].values.astype(float)

# Smooth the prices using a Gaussian filter to reduce noise
smoothed_prices = gaussian_filter1d(close_prices, sigma=2)

# Normalize prices for Sobel
normalized_prices = (smoothed_prices - np.min(smoothed_prices)) / (np.max(smoothed_prices) - np.min(smoothed_prices))

# Apply Sobel filter
price_edges = sobel(normalized_prices)

# Highlight significant changes using a threshold
threshold = 0.05  # Adjust this value as needed
significant_changes = np.abs(price_edges) > threshold

# Convert significant_changes to a Pandas Series with the same index as data
significant_changes = pd.Series(significant_changes, index=data.index)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, normalized_prices, label='Normalized Prices (Smoothed)', color='blue')
plt.plot(data.index, price_edges, label='Sobel Edges (Trend Changes)', color='red', linestyle='--')
plt.scatter(data.index[significant_changes], price_edges[significant_changes], color='green', label='Significant Changes', zorder=5)
plt.title('Gold Prices with Sobel Edge Detection (Smoothed)')
plt.legend()
plt.grid()
plt.show()