import matplotlib.pyplot as plt
import yfinance as yf

def plot_trades(df, trades_df, total_return, initial_capital ,market_df):
    plt.figure(figsize=(14, 7))

    # 下载 S&P 500 数据
    market_df['Close'].fillna(method='ffill', inplace=True)

    # 计算 S&P 500 的持仓曲线（按比例调整）
    market_df['Position Value'] = (market_df['Close'] / market_df['Close'].iloc[0]) * initial_capital

    # 绘制收盘价
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    
    # 绘制买入和卖出信号
    buy_signals = trades_df[trades_df['Action'] == 'Buy']
    sell_signals = trades_df[trades_df['Action'] == 'Sell']

    plt.scatter(buy_signals['Date'], buy_signals['Price'], marker='^', color='g', label='Buy Signal', s=20, zorder=5)
    plt.scatter(sell_signals['Date'], sell_signals['Price'], marker='v', color='r', label='Sell Signal', s=20, zorder=5)

    # 图表标题和标签
    plt.title(f'Total Return: {total_return:.2f} | Initial Capital: ${initial_capital}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.grid()

    # 计算每笔交易的收益并且计算累积收益
    trades_df['Profit'] = trades_df['Price'].diff().fillna(0)  # 每笔交易的利润
    trades_df['Cumulative Profit'] = trades_df['Profit'].cumsum() + initial_capital  # 累计收益基于初始资本

    # 确保buy_signals和sell_signals包括'Cumulative Profit'列
    buy_signals = trades_df[trades_df['Action'] == 'Buy'].copy()
    sell_signals = trades_df[trades_df['Action'] == 'Sell'].copy()

    # 绘制累计收益
    plt.subplot(3, 1, 2)
    plt.plot(trades_df['Date'], trades_df['Cumulative Profit'], label='Strategy Cumulative Profit')
    plt.axhline(initial_capital, color='black', lw=1, ls='--', label='Initial Capital')  # 初始资金线

    # 添加买入和卖出交易符号
    plt.scatter(buy_signals['Date'], buy_signals['Cumulative Profit'], color='g', marker='^', s=20, label='Buy Trades')  # 买入交易标记
    plt.scatter(sell_signals['Date'], sell_signals['Cumulative Profit'], color='r', marker='v', s=20, label='Sell Trades')  # 卖出交易标记

    # 图表标题和标签
    plt.title('Cumulative Profit Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Profit (with initial capital)', fontsize=14)

    plt.legend()
    plt.grid()

    # 绘制与 S&P 500 比较
    plt.subplot(3, 1, 3)
    plt.plot(market_df.index, market_df['Position Value'], label='S&P 500 Position Value', color='blue', linestyle='--')
    plt.plot(trades_df['Date'], trades_df['Cumulative Profit'], label='Strategy Cumulative Profit', color='orange')
    plt.axhline(initial_capital, color='black', lw=1, ls='--', label='Initial Capital')

    plt.title('Comparison with S&P 500', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value (with initial capital)', fontsize=14)
    
    plt.legend()
    plt.grid()

    plt.tight_layout()  # 调整布局
    plt.show()

    # 保存图像
    plt.savefig("trading_results_vs_market.png", format='png')
