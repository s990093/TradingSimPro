import vectorbt as vbt
import numpy as np

# 下载 Apple 股票数据
symbols = ['AAPL']
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

# 定义均线组合
windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])

# 均线策略的买卖信号
entries_ma = fast_ma.ma_crossed_above(slow_ma)
exits_ma = fast_ma.ma_crossed_below(slow_ma)

# 添加 RSI 策略 (14 天)
rsi = vbt.RSI.run(price, window=14)
entries_rsi = rsi.rsi_crossed_below(30)  # RSI 低于 30 时买入
exits_rsi = rsi.rsi_crossed_above(70)  # RSI 高于 70 时卖出

# 组合信号：可以结合均线策略和 RSI 策略
entries_combined = entries_ma & entries_rsi  # 当两个条件都满足时买入
exits_combined = exits_ma & exits_rsi  # 当两个条件都满足时卖出

# 创建投资组合
pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')
pf = vbt.Portfolio.from_signals(price, entries_combined, exits_combined, **pf_kwargs)

# 绘制回测结果的总收益热图
fig = pf.total_return().vbt.heatmap(
    x_level='fast_window', y_level='slow_window', slider_level='symbol', symmetric=True,
    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%'))
)
fig.show()

# 针对特定窗口组合的结果 (例如 10 天和 20 天均线)
pf[(10, 20, 'AAPL')].plot().show()

# 输出统计数据
print(pf.stats())
