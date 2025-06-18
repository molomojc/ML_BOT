import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ComplexSMC:
    def __init__(self, num_particles=1000, transition_std=0.001, obs_std=0.002):
        self.num_particles = num_particles
        self.transition_std = transition_std
        self.obs_std = obs_std
        self.particles = np.random.normal(1.0, 0.005, size=num_particles)  # closer to EUR/USD range
        self.weights = np.ones(num_particles) / num_particles

    def log_gaussian(self, x, mean, std):
        return -0.5 * np.log(2 * np.pi * std**2) - ((x - mean)**2) / (2 * std**2)

    def predict(self):
        noise = np.random.normal(0, self.transition_std, self.num_particles)
        self.particles += noise
        return np.mean(self.particles)

    def update(self, observation):
        log_weights = self.log_gaussian(observation, self.particles, self.obs_std)
        log_weights -= np.max(log_weights)  # For numerical stability
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def run(self, observations):
        self.particles = np.random.normal(observations[0], self.obs_std, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

        predictions = []
        for t in range(1, len(observations)):
            pred = self.predict()
            predictions.append(pred)
            self.update(observations[t])
            self.resample()
        return predictions



def get_btc_hourly_close(period="1d"):
    btc = yf.download("EURUSD=X", period=period, interval= "1h")

    if btc.empty:
        raise ValueError("Failed to fetch BTC data from yfinance. DataFrame is empty.")

    print("Fetched BTC data:\n", btc)

    close_prices = [float(x) for x in btc["Close"].dropna().values]
  #  date_times = btc.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    return close_prices


# === Run Prediction ===
try:
    btc_prices = get_btc_hourly_close()
    if len(btc_prices) < 2:
        print("Not enough price data.")
    else:
        smc = ComplexSMC()
        predictions = smc.run(btc_prices)

        # Collect actuals and predictions
        actuals = btc_prices[1:]  # Since we start predicting from index 1
        preds = predictions

        for i, (pred, actual) in enumerate(zip(preds, actuals)):
            print(f"Time {i+1}: Predicted Close = {pred:.5f}, Actual Close = {actual:.5f}")

        # Calculate accuracy metrics
        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        # === Signal Generation ===
        last_pred = preds[-1]
        last_actual = actuals[-1]
        diff = last_pred - last_actual

        print("\n=== Trading Signal ===")
        print(f"Last Actual Close: {last_actual:.5f}")
        print(f"Predicted Next Close: {last_pred:.5f}")

        print("\n=== Accuracy Metrics ===")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")

except Exception as e:
    print(f"Error: {e}")