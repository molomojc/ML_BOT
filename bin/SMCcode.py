import numpy as np

class ComplexSMC:
    def __init__(self, num_particles=1000, transition_std=1.0, obs_std=2.0):
        self.num_particles = num_particles
        self.transition_std = transition_std
        self.obs_std = obs_std
        self.particles = None
        self.weights = None

    def initialize(self, init_value):
        self.particles = np.random.normal(init_value, self.obs_std, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self):
        noise = np.random.normal(0, self.transition_std, self.num_particles)
        self.particles += noise

    def update(self, observation):
        likelihoods = self.gaussian(observation, self.particles, self.obs_std)
        self.weights *= likelihoods
        self.weights += 1e-300  # avoid divide by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        effective_N = 1.0 / np.sum(np.square(self.weights))
        if effective_N < self.num_particles / 2:
            indices = np.random.choice(
                self.num_particles, self.num_particles, p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def gaussian(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std)**2)

    def run(self, observations):
        predictions = []
        self.initialize(observations[0])
        for obs in observations[1:]:
            self.predict()
            self.update(obs)
            self.resample()
            predictions.append(np.average(self.particles, weights=self.weights))
        return predictions

# Suppose you have closing prices
btc_prices = [84700, 84600, 84550, 84630, 84720, 84810]  # real example would be longer

smc = ComplexSMC(transition_std=20.0, obs_std=30.0)
predictions = smc.run(btc_prices)

for i, pred in enumerate(predictions):
    print(f"Time {i+1}: Predicted Close = {pred:.2f}, Actual Close = {btc_prices[i+1]}")
