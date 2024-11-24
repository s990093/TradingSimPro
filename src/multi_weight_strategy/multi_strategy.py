from typing import List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from .single_strategy import SingleMultiWeightStrategy
import matplotlib.pyplot as plt

class MultiWeightStrategy:
    def __init__(self, stock_id: str, start_date: str, end_date: str, best_solution: np.ndarray, model_path: str,
                 max_positions: int = 5, position_interval_days: int = 5):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.best_solution = best_solution
        self.model_path = model_path
        self.max_positions = max_positions
        self.position_interval_days = position_interval_days
        
        # 只創建一個策略實例
        self.single_strategy = SingleMultiWeightStrategy(
            stock_id=self.stock_id,
            start_date=self.start_date,
            end_date=self.end_date,
            best_solution=self.best_solution,
            model_path=self.model_path
        )

    def run(self) -> Tuple[float, pd.DataFrame]:
        """執行多倉位策略"""
        total_trades_df = pd.DataFrame()
        total_return = 0.0
        
        # 收集所有的 delayed_start 時間
        delayed_starts = []
        for i in range(self.max_positions):
            delayed_start = pd.Timestamp(self.start_date) + pd.Timedelta(days=i * self.position_interval_days)
            if delayed_start >= pd.Timestamp(self.end_date):
                break
            delayed_starts.append(delayed_start)
            
            # 更新策略的開始時間
            self.single_strategy.start_date = delayed_start
            strategy_return, strategy_trades = self.single_strategy.run(show=False)
            
            # 添加策略編號
            strategy_trades = strategy_trades.copy()
            strategy_trades['Strategy_ID'] = i
            
            # 累加收益和交易記錄
            total_return += strategy_return
            print(f"Strategy {i} return: {strategy_return:.2f}")
            total_trades_df = pd.concat([total_trades_df, strategy_trades], ignore_index=True)
        
        # 按時間排序所有交易
        total_trades_df = total_trades_df.sort_values('Trade Date').reset_index(drop=True)
        
        # 重新計算累積收益
        sell_trades = total_trades_df[total_trades_df['Trade Type'].isin(['Stop Loss', 'Take Profit'])]
        if not sell_trades.empty:
            cumulative_returns = sell_trades['Profit Amount'].cumsum()
            sell_indices = sell_trades.index
            total_trades_df.loc[sell_indices, 'Cumulative Return'] = cumulative_returns.values
        
        # 計算策略統計資訊
        total_trades = len(total_trades_df[total_trades_df['Trade Type'] == 'Buy'])
        winning_trades = len(total_trades_df[
            (total_trades_df['Trade Type'].isin(['Stop Loss', 'Take Profit'])) & 
            (total_trades_df['Profit Amount'] > 0)
        ])
        losing_trades = len(total_trades_df[
            (total_trades_df['Trade Type'].isin(['Stop Loss', 'Take Profit'])) & 
            (total_trades_df['Profit Amount'] < 0)
        ])
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # 計算最大回撤
        if not sell_trades.empty:
            cumulative_returns = sell_trades['Cumulative Return']
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max * 100
            max_drawdown = drawdowns.min()
        else:
            max_drawdown = 0
        
        # 顯示策略統計資訊
        print("\n=== 策略統計資訊 ===")
        print(f"股票代碼: {self.stock_id}")
        print(f"交易區間: {self.start_date} 到 {self.end_date}")
        print(f"總收益: {total_return:.2f}")
        print(f"總交易次數: {total_trades}")
        print(f"獲利次數: {winning_trades}")
        print(f"虧損次數: {losing_trades}")
        print(f"勝率: {win_rate:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        
        # 創建三個子圖
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,8), sharex=True)
        
        # 上方子圖：價格圖和交易點
        price_data = self.single_strategy.data_cache.get_data(show=False)
        ax1.plot(price_data.index, price_data['Close'], label='Price', color='blue', alpha=0.6)
        
        # 標記買入點和賣出點
        buy_trades = total_trades_df[total_trades_df['Trade Type'] == 'Buy']
        sell_trades = total_trades_df[total_trades_df['Trade Type'].isin(['Stop Loss', 'Take Profit'])]
        
        ax1.scatter(buy_trades['Trade Date'], buy_trades['Trade Price'], 
                   color='green', marker='^', s=10, label='Buy')
        ax1.scatter(sell_trades['Trade Date'], sell_trades['Trade Price'], 
                   color='red', marker='v', s=10, label='Sell')
        
        # 標記 delayed_start 時間
        for i, start_time in enumerate(delayed_starts):
            # 在價格圖上標記垂直線
            ax1.axvline(x=start_time, color='purple', linestyle='--', alpha=0.5)
        
        ax1.set_title(f'Price Chart with Trade Points for {self.stock_id}')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # 中間子圖：累積收益
        ax2.plot(sell_trades['Trade Date'], sell_trades['Cumulative Return'], 
                color='purple', marker='o', label='Cumulative Return')
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True)
        ax2.legend()
        
        # 下方子圖：持倉數量
        # 計算每日持倉數量
        daily_positions = pd.DataFrame(index=price_data.index)
        daily_positions['Positions'] = 0
        
        # 根據買賣記錄計算持倉
        for _, row in total_trades_df.iterrows():
            if row['Trade Type'] == 'Buy':
                daily_positions.loc[row['Trade Date']:, 'Positions'] += 1
            elif row['Trade Type'] in ['Stop Loss', 'Take Profit']:
                daily_positions.loc[row['Trade Date']:, 'Positions'] -= 1
        
        ax3.plot(daily_positions.index, daily_positions['Positions'], 
                color='orange', label='Number of Positions')
        ax3.set_title('Number of Active Positions')
        ax3.set_xlabel('Trade Date')
        ax3.set_ylabel('Positions')
        ax3.grid(True)
        ax3.legend()
        
        # 設置 y 軸的整數刻度
        ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 旋轉 x 軸標籤
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return total_return, total_trades_df
