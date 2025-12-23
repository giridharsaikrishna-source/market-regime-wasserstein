import pandas as pd
import numpy as np

class Backtester:
    @staticmethod
    def run(data, labels, indices, bear_regime_index, tc=0.002, confirm_days=2):
        results = data.copy()
        regime_map = pd.DataFrame({'Regime': labels}, index=indices)
        results = results.join(regime_map).ffill()
        
        # 1. HYSTERESIS (CONFIRMATION)
        # Prevents "flip-flopping" by requiring 2 days of consistent signal
        results['Is_Bear'] = (results['Regime'] == bear_regime_index).astype(int)
        results['Confirmed_Signal'] = results['Is_Bear'].rolling(window=confirm_days).min().fillna(0)
        
        # 2. VOLATILITY FILTER
        # Only exits if Bear Regime AND Current Vol > Avg Vol
        results['Vol'] = results['Returns'].rolling(21).std()
        results['Vol_Mean'] = results['Vol'].rolling(63).mean()
        # Add this inside the run method in strategy.py
        results['SMA_200'] = results['Close'].rolling(200).mean()
        results['Trend_Down'] = (results['Close'] < results['SMA_200']).astype(int)

        # Refined Signal: Wasserstein Bear + Volatility + Trend
        results['Signal'] = ((results['Confirmed_Signal'] == 1) & 
                            (results['Vol'] > results['Vol_Mean']) &
                            (results['Trend_Down'] == 1)).astype(int)
        
        # 3. 1-DAY EXECUTION LAG
        # Shift the signal so we trade on the NEXT day's price
        results['Action'] = results['Signal'].shift(1).fillna(0)
        
        # Calculate Returns
        results['Strategy_Ret'] = results['Returns']
        results.loc[results['Action'] == 1, 'Strategy_Ret'] = 0
        
        # Apply Transaction Costs
        results['Trade_Occurred'] = results['Action'].diff().fillna(0) != 0
        results['Strategy_Ret_Net'] = results['Strategy_Ret']
        results.loc[results['Trade_Occurred'], 'Strategy_Ret_Net'] -= tc
        
        # Cumulative Wealth
        results['Market_Cum'] = (1 + results['Returns']).cumprod()
        results['Strategy_Cum'] = (1 + results['Strategy_Ret_Net']).cumprod()
        
        metrics = Backtester.calculate_metrics(results)
        return results, metrics

    @staticmethod
    def calculate_metrics(df):
        metrics = {}
        for col, ret_col in [('Market', 'Returns'), ('Strategy', 'Strategy_Ret_Net')]:
            returns = df[ret_col].dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino = (returns.mean() * 252) / downside_std if downside_std != 0 else 0
            
            cum_rets = (1 + returns).cumprod()
            max_dd = ((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()
            annual_ret = (cum_rets.iloc[-1])**(252/len(returns)) - 1 if len(returns) > 0 else 0
            calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
            
            metrics[col] = {
                'Sharpe': round(sharpe, 2),
                'Sortino': round(sortino, 2),
                'Calmar': round(calmar, 2),
                'MaxDD': f"{round(max_dd * 100, 2)}%"
            }
        return metrics