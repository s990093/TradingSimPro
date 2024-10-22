import pandas as pd
import numpy as np
import yfinance as yf

# 获取股票数据
def get_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start, end=end)['Close']
    return pd.DataFrame(data)

# 定义策略信号（示例）
def strategy_signals(data, ma_length):
    signals = {}
    signals['A'] = (data['AAPL'].rolling(window=ma_length).mean().shift(1) < data['AAPL']).astype(int)  # 策略 A
    signals['B'] = (data['MSFT'].rolling(window=ma_length).mean().shift(1) < data['MSFT']).astype(int)  # 策略 B
    signals['C'] = (data['GOOGL'].rolling(window=ma_length).mean().shift(1) < data['GOOGL']).astype(int)  # 策略 C
    return pd.DataFrame(signals)

# 计算组合信号
def combine_signals(signals, logic='AND'):
    if logic == 'AND':
        return signals.prod(axis=1)  # 所有策略同时发出信号
    elif logic == 'OR':
        return (signals.sum(axis=1) > 0).astype(int)  # 任意策略发出信号

# 评估适应度
def evaluate_fitness(ma_length, data):
    signals = strategy_signals(data, ma_length)
    combined_signal = combine_signals(signals, logic='AND')
    returns = data.pct_change().shift(-1)
    strategy_returns = combined_signal * returns
    total_return = strategy_returns.sum()
    return total_return

# ABC算法实现
def abc_algorithm(data, num_food_sources=10, iterations=100):
    # 初始化蜜蜂的位置
    food_sources = np.random.randint(5, 50, size=(num_food_sources, 1))  # 假设移动平均线的长度范围
    fitness = np.zeros(num_food_sources)

    for iteration in range(iterations):
        for i in range(num_food_sources):
                fitness[i] = evaluate_fitness(food_sources[i, 0], data)

        # 更新蜜蜂位置
        for i in range(num_food_sources):
            new_position = food_sources[i, 0] + np.random.randint(-1, 2)  # 随机调整
            if 5 <= new_position <= 50:
                new_fitness = evaluate_fitness(new_position, data)
                if new_fitness > fitness[i]:
                    food_sources[i, 0] = new_position

    # 返回最佳参数
    best_index = np.argmax(fitness)
    return food_sources[best_index, 0]

# 示例调用
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = get_stock_data(tickers, start='2020-01-01', end='2023-01-01')

# 使用 ABC 算法找到最佳移动平均线长度
best_ma_length = abc_algorithm(data)
print("最佳移动平均线长度:", best_ma_length)

# 使用最佳参数计算最终信号
final_signals = strategy_signals(data, best_ma_length)
final_combined_signal = combine_signals(final_signals, logic='AND')

# 计算最终收益
final_returns = data.pct_change().shift(-1) * final_combined_signal
print("最终收益:", final_returns.sum())
