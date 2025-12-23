# Market Regime Discovery: An Optimal Transport Approach
### *Unsupervised Learning of Financial Distributions via Wasserstein-1 Distance*

## 📌 Project Overview
This project implements a non-parametric market regime detection system using **1-Wasserstein Distance** (Earth Mover's Distance). Unlike traditional Moving Average or HMM models, this approach clusters the "shape" of return distributions, making it robust to the non-Gaussian nature of financial markets.

## 🚀 Key Features
* **Wasserstein K-Means:** Custom engine using Optimal Transport to identify structural shifts in market behavior.
* **Walk-Forward Validation (WFV):** Industry-standard testing that retrains the model every 6 months to eliminate look-ahead bias.
* **Market Frictions:** Conservative backtesting accounting for **1-day execution lag** and **0.2% slippage/taxes**.
* **Multi-Asset Benchmarking:** Comparative study across NIFTY 50 (Trending) vs. Bank Nifty (Mean-reverting).

## 📈 Performance Summary

| Asset | Strategy Sharpe | Market Sharpe | Max Drawdown (Strat) | Max Drawdown (Mkt) |
| :--- | :--- | :--- | :--- | :--- |
| **Nifty 50** | **1.14** | 0.71 | **-23.79%** | -40.04% |
| **Bank Nifty** | 1.02 | 1.07 | -25.33% | -22.40% |

### **Strategy Performance**
![Walk-Forward Equity Curve](assets/Walk-Forward%20Equity%20curve.png)

### Research Insight
The model significantly outperformed on the **Nifty 50**, nearly halving the Maximum Drawdown while improving risk-adjusted returns (Sharpe Ratio). For **Bank Nifty**, the results highlight the challenges of regime-switching in high-noise, mean-reverting environments.

## 🧠 Methodology
1.  **Engine:** Wasserstein-1 distance measures the work required to transform one return distribution into another.
2.  **Filters:** Uses a combination of Volatility Z-scores and 200-day Moving Averages to confirm signals.
3.  **Stability:** Implements `n_init` multi-start clustering to ensure stable centroids across time-steps.

### **Learnt Regime Distributions**
![Interpretability Plot](assets/Interpretability%20plot.png)

## 🛠️ Tech Stack
* **Python 3.12**
* **NumPy / Pandas** (Data Processing)
* **yFinance** (Market Data)
* **Matplotlib** (Visualization)

## 📁 Project Structure
* `data_loader.py`: Cached data retrieval.
* `engine.py`: Wasserstein clustering logic.
* `strategy.py`: Backtesting and metric calculations.
* `walk_forward.py`: Main orchestrator for out-of-sample testing.