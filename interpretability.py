import numpy as np
import matplotlib.pyplot as plt
from data_loader import MarketData
from engine import WassersteinClustering

# 1. Load and Train on a large sample
loader = MarketData(ticker="^NSEI", window_size=21)
raw_data, windows, indices = loader.get_processed_data()

model = WassersteinClustering(n_regimes=2, n_init=10)
model.fit(windows)

# 2. Identify Bear/Bull
centroids = model.get_centroids()
means = [np.mean(c) for c in centroids]
bear_idx = np.argmin(means)
bull_idx = 1 - bear_idx

# 3. Plot the "Shapes" of the Regimes
plt.figure(figsize=(10, 6))

# We use the sorted returns to show the cumulative distribution or a histogram
plt.plot(centroids[bull_idx], np.linspace(0, 1, len(centroids[bull_idx])), 
         label='Bull Regime (Stable/Positive)', color='green', lw=3)
plt.plot(centroids[bear_idx], np.linspace(0, 1, len(centroids[bear_idx])), 
         label='Bear Regime (Fat-Tailed/Negative)', color='red', lw=3)

plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.title("Wasserstein Interpretability: Learnt Market 'Shapes'")
plt.xlabel("Log Returns")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()