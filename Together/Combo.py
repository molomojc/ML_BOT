#import relevant packages
import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta

# a function to fetch data and flatten it
def fetch_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    #for sake of accuracy ensure the 
    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    # Lowercase column names for consistency
    df.columns = [col.lower() for col in df.columns]
    return df

#function to train the model
def train_model():
    
