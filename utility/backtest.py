def run_backtest(weights, df):
    combined_signal = (
        weights['ma_signal'] * df['ma_signal'] +
        weights['rsi_signal'] * df['rsi_signal'] +
        weights['macd_signal'] * df['macd_signal'] +
        weights['bb_signal'] * df['bb_signal'] +
        weights['momentum_signal'] * df['momentum_signal'] +
        weights['stochastic_signal'] * df['stochastic_signal']
    ) / sum(weights.values())
    
    df['combined_positions'] = combined_signal.diff()

    # 計算買賣信號
    buy_signals = df['combined_positions'].apply(lambda x: 1 if x > 0.1 else 0)
    sell_signals = df['combined_positions'].apply(lambda x: 1 if x < -0.1 else 0)

    # 計算收益
    df['returns'] = df['Adj Close'].pct_change().fillna(0)
    df['strategy_returns'] = buy_signals.shift(1) * df['returns']  # 根據信號計算策略收益
    cumulative_returns = (1 + df['strategy_returns']).cumprod() - 1
    final_return = cumulative_returns.iloc[-1]

    return final_return

# 回測不同的權重組合
for i in range(10):  # 假設我們跑10次回測
    new_weights = {
        'ma_signal': np.random.random(),
        'rsi_signal': np.random.random(),
        'macd_signal': np.random.random(),
        'bb_signal': np.random.random(),
        'momentum_signal': np.random.random(),
        'stochastic_signal': np.random.random()
    }
    
    # 正規化權重，使它們的總和為1
    total_weight = sum(new_weights.values())
    for key in new_weights:
        new_weights[key] /= total_weight
    
    final_return = run_backtest(new_weights)
    
    # 保存回測結果
    backtest_results = backtest_results.append({
        'ma_weight': new_weights['ma_signal'],
        'rsi_weight': new_weights['rsi_signal'],
        'macd_weight': new_weights['macd_signal'],
        'bb_weight': new_weights['bb_signal'],
        'momentum_weight': new_weights['momentum_signal'],
        'stochastic_weight': new_weights['stochastic_signal'],
        'final_returns': final_return
    }, ignore_index=True)
