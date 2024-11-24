from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from backtest.base_strategy_backtest import BaseStrategyBacktest
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel

class EnhancedStrategyBacktest(BaseStrategyBacktest):
    def __init__(self, stock_id: str, start_date: str, end_date: str, strategy, show_progress: bool = True):
        super().__init__(stock_id, start_date, end_date, strategy)
        self.show_progress = show_progress 
        
    def run(self):
        df = self.data_cache.get_data(self.show_progress)
        self.strategy.reset()
        in_position = False
        positions = []  # 用於記錄持倉變化
        times = []  # 用於記錄時間
        stock_prices = []  # List to record stock prices
        buy_points = []  # List to record buy points
        risk_reward = []  # List to record risk-reward values

        # Calculate daily price changes (not using percentage)
        df['price_change_daily'] = df['Close'] - df['Close'].shift(1)
        close_prices = df['Close'].values  # Access the column once
        price_changes = df['price_change_daily'].values  # Cache price changes

        
        if self.show_progress:  # 根据参数决定是否显示进度条
            self.console.status("[bold green]Running backtest...")

            with Progress() as progress:  # Use Progress class
                task = progress.add_task("Processing trades", total=len(df) - 1)  # Add task
                
                for idx in range(1, len(df)):
                    current_price = close_prices[idx]  
                    price_change = price_changes[idx] 
                    
                    if not in_position:
                        if price_change > 0: 
                            entry_price = current_price
                            self.strategy.enter_position(entry_price, df.index[idx])
                            in_position = True
                            buy_points.append((df.index[idx], entry_price))  # Record buy point
                    else:
                        action = self.strategy.update(current_price, df.index[idx])
                        if action in ['STOP_LOSS', 'TAKE_PROFIT']:
                            exit_price = current_price
                            risk = entry_price - self.strategy.stop_loss_pct  # Assuming stop_loss is defined
                            reward = exit_price - entry_price
                            risk_reward.append((df.index[idx], risk, reward))  # Record risk and reward
                            in_position = False
                    
                    # Record positions, time, and stock prices
                    positions.append(self.strategy.current_position)  # Assuming there is a current_position attribute
                    times.append(df.index[idx])  # Record time
                    stock_prices.append(current_price)  # Record stock price
                    
                    progress.update(task, advance=1)  #
        else:
            for idx in range(1, len(df)):
                current_price = close_prices[idx]  # Use cached value
                price_change = price_changes[idx]  # Use cached value
                
                if not in_position:
                    if price_change > 0: 
                        entry_price = current_price
                        self.strategy.enter_position(entry_price, df.index[idx])
                        in_position = True
                        buy_points.append((df.index[idx], entry_price))  # Record buy point
                else:
                    action = self.strategy.update(current_price, df.index[idx])
                    if action in ['STOP_LOSS', 'TAKE_PROFIT']:
                        exit_price = current_price
                        risk = entry_price - self.strategy.stop_loss_pct  # Assuming stop_loss is defined
                        reward = exit_price - entry_price
                        risk_reward.append((df.index[idx], risk, reward))  # Record risk and reward
                        in_position = False
                
                # Record positions, time, and stock prices
                positions.append(self.strategy.current_position)  # Assuming there is a current_position attribute
                times.append(df.index[idx])  # Record time
                stock_prices.append(current_price)  # Record stock price

        self._calculate_metrics()
        
        if self.show_progress:
            self._show_res()
            self.display_charts(times, positions, stock_prices, buy_points, risk_reward)
                   
    def _calculate_metrics(self):
        trades_df = pd.DataFrame(self.strategy.trades_history)
        if trades_df.empty:
            return

        # Initialize variables
        realized_pnl = 0.0
        position_cost = 0.0
        current_position = 0
        trades = []
        trade_times = []  # List to record the corresponding times for unrealized P&L

        for _, row in trades_df.iterrows():
            price = row['price']
            action = row['action']
            time = row['timestamp'] 

      
            if action.startswith("ENTER"):  # Process "ENTER" actions
                current_position = row['position']
                position_cost = price * current_position
            elif action.startswith("ML_ADD") or action == "ADD":  # Process "ADD" actions
                added_units = row['position'] - current_position
                position_cost += price * added_units
                current_position = row['position']
            elif any(action.startswith(x) for x in ["ML_REDUCE", "ML_EXIT", "REDUCE", "STOP_LOSS"]):  # Process "REDUCE/EXIT" actions
                reduced_units = current_position - row['position']
                avg_cost = position_cost / current_position if current_position > 0 else 0
                trade_pnl = float((price - avg_cost) * reduced_units)
                trades.append(trade_pnl)
                trade_times.append(time)
                realized_pnl += trade_pnl

                # Update position cost for remaining position
                if row['position'] > 0:
                    position_cost = (position_cost / current_position) * row['position']
                else:
                    position_cost = 0
                current_position = row['position']

        # Calculate metrics using trades list
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t < 0]
        
        self.results = {
            'total_profit': float(sum(trades)), 
            'total_trades': len(trades),
            'trades': trades,
            'trade_times': trade_times,
            'win_rate': float(sum(1 for t in trades if t > 0) / len(trades)) if trades else 0.0,  # 確保 trades 不為空
            'profit_factor': float(abs(sum(winning_trades) / sum(losing_trades))) if losing_trades and sum(losing_trades) != 0 else float('inf'),  # 確保 losing_trades 非空且分母不為零
            'win_loss_ratio': float((sum(winning_trades) / len(winning_trades)) / abs(sum(losing_trades) / len(losing_trades))) 
                            if winning_trades and losing_trades and len(losing_trades) != 0 else float('inf'),  # 確保分母不為零
            'max_drawdown': float(min(trades) if trades else 0),  # 確保 trades 不為空
            'trade_counts': self._calculate_trade_counts(trades)
        }
