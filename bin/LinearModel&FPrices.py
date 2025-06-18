import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump

# 1. Fetch BTC-USD price data
def get_btc_data(period="7d", interval="1h"):
    df = yf.download("BTC-USD", period=period, interval=interval)
    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Lowercase column names for consistency
    df.columns = [col.lower() for col in df.columns]

    df.dropna(inplace=True)
    return df

# 2. Calculate technical indicators
def compute_indicators(df):
    df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df.dropna()

# 3. Prepare features and target
def prepare_dataset(df):
    df['target'] = df['close'].shift(-1)  # Predict next closing price
    features = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'macd_diff']
    df = df.dropna()
    X = df[features]
    y = df['target']
    return X, y

# 4. Train multiple regression model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    return model, X_test, y_test, y_pred


# 5. Run the pipeline
df = get_btc_data()
df = compute_indicators(df)
X, y = prepare_dataset(df)
model, X_test, y_test, y_pred = train_model(X, y)
dump(model, 'regression_model.pkl')

print(df)
# 6. Show most recent prediction vs actual
latest = pd.DataFrame({'Predicted': y_pred[-5:], 'Actual': y_test[-5:].values}, index=y_test.index[-5:])
print("\nRecent Predictions:")
print(latest)