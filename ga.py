import pandas as pd

from utility.helper.stock_data_cache import StockDataCache
from ENV import Environment

# 假设这是你生成的最佳交易 DataFrame
best_trades_df_read = pd.read_pickle('best_trades.pkl')

print(best_trades_df_read)

# 获取股票数据
df_data = StockDataCache(Environment.target_stock, Environment.start_date, Environment.end_date).get_data()

# 计算可交易张数
def calculate_trade_volume(account_balance, risk_percentage, price):
    risk_amount = account_balance * (risk_percentage / 100)
    return int(risk_amount / price)

# 调整交易张数
def adjust_trade_volumes(best_trades_df, df_data, account_balance, risk_percentage):
    adjusted_volumes = []

    for index, row in best_trades_df.iterrows():
        buy_price = row['Price']  # 使用 'Price' 列
        
        # 计算可交易的张数
        volume = calculate_trade_volume(account_balance, risk_percentage, buy_price)
        
        # 可交易张数大于0则存入列表
        adjusted_volumes.append(volume if volume > 0 else 0)

    best_trades_df['Adjusted_Volume'] = adjusted_volumes
    return best_trades_df

# 评估盈利能力
def evaluate_profitability(best_trades_df, df_data, profit_threshold=0.05):
    profitable_trades = []
    
    for index, row in best_trades_df.iterrows():
        buy_price = row['Price']
        buy_date = row['Date']
        
        # 找到买入后的30天内的价格
        future_prices = df_data[df_data.index > buy_date].head(30)  # 使用索引进行过滤
        
        if not future_prices.empty:
            max_future_price = future_prices['Close'].max()  # 使用 'Close' 列
            
            # 判断是否盈利超过预设阈值
            if (max_future_price - buy_price) / buy_price >= profit_threshold:
                profitable_trades.append(row)

    return pd.DataFrame(profitable_trades)

# 设置账户余额和风险比例
account_balance = 10000  # 假设账户余额为10000
risk_percentage = 1  # 风险控制在1%

# 调整交易张数
# adjust_trades_df = adjust_trade_volumes(best_trades_df_read, df_data, account_balance, risk_percentage)
trades_df = best_trades_df_read.copy()  # Use the adjusted trades DataFrame

