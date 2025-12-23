import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import MarketData
from engine import WassersteinClustering
from strategy import Backtester

# --- CONFIGURATION ---
TICKER = "^NSEI"  # Try "^NSEBANK" or "GC=F" for Gold next!
TRAIN_SIZE = 252 * 4  # 4 Years to capture various market cycles
STEP_SIZE = 126       # Retrain every 6 months (more adaptive)
WINDOW_SIZE = 21

# 1. Load Data
loader = MarketData(ticker=TICKER, window_size=WINDOW_SIZE)
raw_data, windows, indices = loader.get_processed_data()

all_predictions = []
test_indices = []

# 2. Walk-Forward Loop
print(f"Running Industry-Grade Walk-Forward for {TICKER}...")
for i in range(TRAIN_SIZE, len(windows), STEP_SIZE):
    train_windows = windows[i - TRAIN_SIZE : i]
    end_idx = min(i + STEP_SIZE, len(windows))
    test_windows = windows[i : end_idx]
    
    # Train
    model = WassersteinClustering(n_regimes=2, n_init=10) # Higher n_init for stability
    model.fit(train_windows)
    
    # Identify Bear (Min Mean)
    bear_regime = np.argmin([np.mean(c) for c in model.centroids])
    
    # Predict & Map (1 = Bear/Cash, 0 = Bull/Invested)
    preds = model.predict(test_windows)
    binary_preds = [1 if p == bear_regime else 0 for p in preds]
    
    all_predictions.extend(binary_preds)
    test_indices.extend(indices[i : end_idx])

# 3. Backtest
# wf_results, wf_metrics = Backtester.run(
#     raw_data.loc[test_indices[0]:], 
#     all_predictions, 
#     test_indices, 
#     bear_regime_index=1, 
#     tc=0.002 # 0.2% Slippage + Taxes
# )

# --- SENSITIVITY ANALYSIS LOOP ---
best_calmar = -np.inf
best_cfg = {}

print("\nSearching for Optimal Asset Configuration...")
for c_days in [1, 2, 3]:
    for tc_val in [0.001, 0.002]: # Testing lower vs higher slippage impact
        wf_results, wf_metrics = Backtester.run(
            raw_data.loc[test_indices[0]:], 
            all_predictions, 
            test_indices, 
            bear_regime_index=1, 
            tc=tc_val,
            confirm_days=c_days
        )
        
        current_calmar = wf_metrics['Strategy']['Calmar']
        if current_calmar > best_calmar:
            best_calmar = current_calmar
            best_cfg = {'confirm_days': c_days, 'tc': tc_val, 'metrics': wf_metrics}

print(f"\nBest Config for {TICKER}: Confirm Days={best_cfg['confirm_days']}, TC={best_cfg['tc']}")
print(f"Optimized Calmar: {best_cfg['metrics']['Strategy']['Calmar']}")

# 4. Output Industry Metrics
print("\n" + "="*30)
print(f"RESULTS FOR {TICKER}")
print("="*30)
for k, v in wf_metrics.items():
    print(f"{k.upper():<10} | Sharpe: {v['Sharpe']:>5} | Sortino: {v['Sortino']:>5} | Calmar: {v['Calmar']:>5} | MaxDD: {v['MaxDD']}")

# 5. Professional Multi-Panel Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Top Plot: Equity Curves
ax1.plot(wf_results['Market_Cum'], label='Benchmark (Nifty)', color='gray', alpha=0.4)
ax1.plot(wf_results['Strategy_Cum'], label='Wasserstein Strategy', color='blue', lw=2)
# Change this line in the plotting section of walk_forward.py:
ax1.fill_between(wf_results.index, wf_results['Strategy_Cum'].min(), wf_results['Strategy_Cum'].max(), 
                 where=(wf_results['Action'] == 1), color='red', alpha=0.15, label='In Cash (Bear Signal)')
ax1.set_title(f"Industry-Grade Backtest: {TICKER}")
ax1.legend()

# Bottom Plot: Drawdown Profile (The "Stress Test")
market_dd = (wf_results['Market_Cum'] - wf_results['Market_Cum'].cummax()) / wf_results['Market_Cum'].cummax()
strat_dd = (wf_results['Strategy_Cum'] - wf_results['Strategy_Cum'].cummax()) / wf_results['Strategy_Cum'].cummax()
ax2.fill_between(wf_results.index, market_dd, 0, color='gray', alpha=0.2, label='Market DD')
ax2.fill_between(wf_results.index, strat_dd, 0, color='red', alpha=0.4, label='Strategy DD')
ax2.set_title("Drawdown Profile (Risk Exposure)")
ax2.legend()

plt.tight_layout()
plt.show()