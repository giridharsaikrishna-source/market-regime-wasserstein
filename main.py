import numpy as np
import matplotlib.pyplot as plt
from data_loader import MarketData
from engine import WassersteinClustering
from strategy import Backtester

# 1. Load Data
loader = MarketData(ticker="^NSEBANK", window_size=21)
raw_data, windows, indices = loader.get_processed_data()

# 2. Cluster - Strictly 2 Regimes
model = WassersteinClustering(n_regimes=2, n_init=10)
labels = model.fit(windows)

# 3. Identify Bear vs Bull
# The regime with the lower mean of sorted returns is the Bear
regime_means = [np.mean(c) for c in model.centroids]
bear_regime = np.argmin(regime_means)
bull_regime = 1 - bear_regime

print(f"Clustering Complete.")
print(f"Regime {bull_regime}: Bull (Growth)")
print(f"Regime {bear_regime}: Bear (Crash/Volatility)")

# 4. Backtest with 0.1% Transaction Cost
tester = Backtester()
final_results, perf_metrics = tester.run(raw_data, labels, indices, bear_regime, tc=0.001)

# 5. Output Results
print("\n--- Industry Standard Metrics ---")
for model_name, m in perf_metrics.items():
    print(f"{model_name.upper()}: Sharpe={m['Sharpe Ratio']}, MaxDD={m['Max Drawdown']}")

# 6. Plotting
plt.figure(figsize=(14, 7))
plt.plot(final_results['Market_Cum'], label='NIFTY 50 Buy & Hold', color='gray', alpha=0.5)
plt.plot(final_results['Strategy_Cum'], label='Wasserstein Regime Strategy (Net)', color='blue', lw=2)

# Shade the identified Bear periods
plt.fill_between(final_results.index, final_results['Strategy_Cum'].min(), final_results['Strategy_Cum'].max(), 
                 where=(final_results['Regime'] == bear_regime), color='red', alpha=0.15, label='Detected Bear Regime')

plt.title("NIFTY 50: 2-Regime Wasserstein Optimal Transport Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()