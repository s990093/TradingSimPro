import itertools
from mult_strategy.strategies import create_strategies
from multi_weight_strategy.data_type import TradeRecord, TradeType, TradingResults
from multi_weight_strategy.models.TradeClassifier import TradeClassifier
from utils.stock_data_cache import StockDataCache
from rich.console import Console
import numpy as np
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional
from rich.console import Console
from alog.pso import pso
import json
import os


class SingleMultiWeightStrategy:
    def __init__(self, stock_id, start_date, end_date, best_solution : np.ndarray, model_path):
        self.classifier = TradeClassifier(model_path)
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.best_solution = best_solution
        self.data_cache = StockDataCache(stock_id, start_date, end_date)
        self.console = Console()
        # 新增預設參數
        self.trading_params = {
            'profit_prob_threshold': 0.75,    # prediction_probs閾值
            'rsi_lower': 50,                  # RSI下限
            'rsi_upper': 70,                  # RSI上限
            'volume_ratio': 1.2,              # 成交量比率
            'prediction_threshold': 4,         # 預測分數閾值
            'stop_loss_pct': 0.008,           # 止損百分比
            'take_profit_pct': 0.025,         # 獲利目標百分比
            'dynamic_sl_rsi': 40,             # 動態止損RSI閾值
            'dynamic_sl_prob': 0.8,           # 動態止損信心度閾值
            'dynamic_tp_rsi': 65,             # 動態獲利RSI閾值
            'dynamic_tp_prob': 0.6,           # 動態獲利信心度閾值
        }
        
        self.cache = {} 
        self.optimized_params_path = f"res/{self.stock_id}/multi_weight_strategy/single_optimized_params.json"
                

    def create_trading_conditions(self, params, predictions, prediction_probs, df):
        """根據優化後的參數創建交易條件"""
        return (
            (predictions >= params['prediction_threshold']) &
            (prediction_probs[:, 1] >= params['profit_prob_threshold']) &
            (df['Close'] > df['MA5']) &
            (df['MA5'] > df['MA10']) &
            (df['RSI'] > params['rsi_lower']) & 
            (df['RSI'] < params['rsi_upper']) &
            (df['Volume'] > df['Volume_MA5'] * params['volume_ratio'])
        )

    def save_parameters(self):
        """保存最佳化參數到文件"""
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(self.optimized_params_path), exist_ok=True)
        
        with open(self.optimized_params_path, 'w') as f:
            json.dump(self.trading_params, f)
        self.console.print(f"Parameters saved to {self.optimized_params_path}")

    def load_parameters(self):
        """從文件獲取最佳化參數"""
        if os.path.exists(self.optimized_params_path):
            with open(self.optimized_params_path, 'r') as f:
                self.trading_params = json.load(f)
            self.console.print(f"Parameters loaded from {self.optimized_params_path}")
        else:
            self.console.print(f"File {self.optimized_params_path} does not exist. Using default parameters.")

    def optimize_parameters(self):
        """使用PSO優化交易參數"""
        def objective_function(x):
            params = {
                'profit_prob_threshold': x[0],
                'rsi_lower': x[1],
                'rsi_upper': x[2],
                'volume_ratio': x[3],
                'prediction_threshold': x[4],
                'stop_loss_pct': x[5],
                'take_profit_pct': x[6],
                'dynamic_sl_rsi': x[7],
                'dynamic_sl_prob': x[8],
                'dynamic_tp_rsi': x[9],
                'dynamic_tp_prob': x[10]
            }
            
            total_return, _ = self.backtest_with_params(params)
            return -total_return  

        # 定義參數範圍
        lb = [0.5,  30, 60, 1.0, 3, 0.005, 0.015, 30, 0.6, 60, 0.5]    # 限
        ub = [0.95, 50, 80, 2.0, 5, 0.015, 0.035, 45, 0.9, 75, 0.7]    # 上限

        # 運行PSO
        xopt, fopt = pso(
            objective_function,
            lb,
            ub,
            swarmsize=150,  
            maxiter=20,  
            omega=0.5,    
            phip=1.5,  
            phig=1.5,     
            debug=True
        )

        # 更新最佳參數
        self.trading_params = {
            'profit_prob_threshold': xopt[0],
            'rsi_lower': xopt[1],
            'rsi_upper': xopt[2],
            'volume_ratio': xopt[3],
            'prediction_threshold': xopt[4],
            'stop_loss_pct': xopt[5],
            'take_profit_pct': xopt[6],
            'dynamic_sl_rsi': xopt[7],
            'dynamic_sl_prob': xopt[8],
            'dynamic_tp_rsi': xopt[9],
            'dynamic_tp_prob': xopt[10]
        }

        # 保存最佳化參數
        self.save_parameters()


        return self.trading_params

    def backtest_with_params(self, params):
        """使用給定參數進行回測"""
        # Get or calculate basic data
        if 'base_data' not in self.cache:
            df = self.data_cache.get_data(show=False)
            
            # Calculate technical indicators
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate volume indicators
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            
            # Cache the results
            self.cache['base_data'] = df
        else:
            df = self.cache['base_data']

        # Get or calculate ML predictions
        if 'ml_predictions' not in self.cache:
            features = self.classifier.calculate_technical_features(df)
            predictions = self.classifier.predict(features)
            prediction_probs = self.classifier.predict_proba(features)
            self.cache['ml_predictions'] = (predictions, prediction_probs)
        else:
            predictions, prediction_probs = self.cache['ml_predictions']

        # Get or calculate strategy signals
        if 'strategy_signals' not in self.cache:
            strategy_manager = create_strategies()
            df_strategy = strategy_manager.apply_all_strategies(df)
            signal_columns = strategy_manager.get_signal_columns()
            df_strategy_np = df_strategy.to_numpy()
            df_data_open = df.Open.to_numpy()
            self.cache['strategy_signals'] = (signal_columns, df_strategy_np, df_data_open)
        else:
            signal_columns, df_strategy_np, df_data_open = self.cache['strategy_signals']

        # Create trading conditions with current parameters
        trading_conditions = self.create_trading_conditions(
            params, predictions, prediction_probs, df
        )
        
        total_return, trades_df = self.ml_fitness(
            self.best_solution, 
            1.0, 
            1.0, 
            df_strategy_np, 
            df_data_open, 
            signal_columns,
            trading_conditions=trading_conditions
        )

        return total_return, trades_df
    

    def run(self, show = True):        
        self.load_parameters()

        total_return, trades_df = self.backtest_with_params(self.trading_params)
        
        total_return, trades_df = self._calculate_metrics(trades_df, total_return, show) 
        
        return total_return, trades_df
    

    def ml_fitness(self, weights: np.ndarray, buy_threshold: float, sell_threshold: float, 
            df_strategy_signals: np.ndarray, close_prices: np.ndarray, 
            signal_columns: list, trading_conditions: Optional[pd.Series] = None,
            model=None) -> Tuple[float, pd.DataFrame]:

        # Calculate weighted signals
        weighted_signals = np.dot(df_strategy_signals, weights)
        
        # Generate trade signals based on thresholds
        buy_signals = weighted_signals > buy_threshold
        sell_signals = weighted_signals < -sell_threshold
        
        # Create trades array
        trades = np.zeros(len(close_prices))
        trades[buy_signals] = 1
        trades[sell_signals] = -1
        
        # Get cached data
        df = self.cache['base_data']
        _, prediction_probs = self.cache['ml_predictions']
        
        # 使用優化後的參數
        stop_loss_pct = self.trading_params['stop_loss_pct']
        take_profit_pct = self.trading_params['take_profit_pct']
        
        # 4. 交易執行邏輯
        total_return = 0.0
        holding = False
        buy_price = 0.0
        trade_records = []
        
        # 預先計算 Volume ratio 以提高效率
        volume_ratio = df['Volume'] / df['Volume_MA5']
        
        for i in range(len(close_prices)):
            current_date = df.index[i]
            
            if not np.isnan(df['MA5'].iloc[i]):  # 確保技術指標已經計算完成
                if trading_conditions.iloc[i] and not holding:
                    # 計算預期獲利百分比
                    expected_profit_pct = (df['Close'].iloc[i] * (1 + take_profit_pct) - df['Close'].iloc[i]) / df['Close'].iloc[i]
                    
                    # 只有當預期獲利百分比大於等於設定值時才交易
                    buy_price = close_prices[i]
                    holding = True
                    trade_records.append(TradeRecord.create_buy(
                        date=current_date,
                        price=buy_price,
                        cum_return=total_return,
                        rsi=df['RSI'].iloc[i],
                        vol_ratio=volume_ratio.iloc[i]
                    ))
                
                elif holding:
                    current_price = close_prices[i]
                    current_rsi = df['RSI'].iloc[i]
                    current_prob = prediction_probs[i, 1]
                    
                    # 使用優化後的動態止損參數
                    current_stop_loss = stop_loss_pct
                    if (current_rsi < self.trading_params['dynamic_sl_rsi'] or 
                        current_prob < self.trading_params['dynamic_sl_prob']):  
                        current_stop_loss = stop_loss_pct * 0.5
                    
                    if current_price <= buy_price * (1 - current_stop_loss):
                        profit = current_price - buy_price
                        total_return += profit
                        holding = False
                        trade_records.append(TradeRecord.create_stop_loss(
                            date=current_date,
                            price=current_price,
                            trigger_price=current_price,
                            cum_return=total_return,
                            profit=profit,
                            rsi=current_rsi,
                            vol_ratio=volume_ratio.iloc[i]
                        ))
                    
                    # 動態獲利了結
                    elif (current_price >= buy_price * (1 + take_profit_pct) or 
                          (current_rsi > self.trading_params['dynamic_tp_rsi'] and current_price > buy_price) or
                          (current_prob < self.trading_params['dynamic_tp_prob'] and current_price > buy_price)):
                        profit = current_price - buy_price
                        total_return += profit
                        holding = False
                        trade_records.append(TradeRecord.create_take_profit(
                            date=current_date,
                            price=current_price,
                            cum_return=total_return,
                            profit=profit,
                            rsi=current_rsi,
                            vol_ratio=volume_ratio.iloc[i]
                        ))
        
        # 修改 DataFrame 創建方式
        trades_df = pd.DataFrame([trade.to_dict() for trade in trade_records])
        # self._calculate_metrics(trades_df, total_return)
        return total_return, trades_df


    def _calculate_metrics(self, trades_df, total_return, show: bool = True):
        if not trades_df.empty:
            completed_trades = trades_df[trades_df['Trade Type'] != 'Buy']

            
            # 繪製交易結果圖表
            if show:
                self.plot_trading_results(trades_df, self.data_cache.get_data(show=False))
            
            # Calculate trading statistics
            trades = completed_trades['Profit Amount'].tolist()
            winning_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            
            # Calculate max consecutive wins/losses
            trade_results = [1 if t > 0 else 0 for t in trades]
            max_consecutive_wins = max((sum(1 for _ in g) for k, g in itertools.groupby(trade_results) if k), default=0)
            max_consecutive_losses = max((sum(1 for _ in g) for k, g in itertools.groupby(trade_results) if not k), default=0)
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(trades)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = rolling_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

            # Store results
            self.results = TradingResults(
                total_profit=float(sum(trades)),
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=float(len(winning_trades) / len(trades)) if trades else 0.0,
                avg_win=float(sum(winning_trades) / len(winning_trades)) if winning_trades else 0.0,
                avg_loss=float(sum(losing_trades) / len(losing_trades)) if losing_trades else 0.0,
                profit_factor=float(abs(sum(winning_trades) / sum(losing_trades))) if losing_trades and sum(losing_trades) != 0 else float('inf'),
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                max_drawdown=float(max_drawdown),
                largest_win=float(max(winning_trades)) if winning_trades else 0.0,
                largest_loss=float(min(losing_trades)) if losing_trades else 0.0,
                trade_counts=len(trades)
            )
            
            if show:
                self.results.show_results()
        
            # self.results.show_results()
        return total_return, trades_df

    def plot_trading_results(self, trades_df: pd.DataFrame, df: pd.DataFrame):
        """
        Visualize trading results
        """
        # 使用默認樣式或其他內建樣式
        plt.style.use('default')  # 或使用 'classic', 'bmh', 'ggplot' 等
        
        # 設置中文字體（如果需要）
        plt.rcParams['font.sans-serif'] = ['Arial']  # 使用 Arial 字體
        plt.rcParams['axes.unicode_minus'] = False   # 正確顯示負號
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Cumulative Returns
        ax1 = plt.subplot(2, 2, 1)
        trades_df.plot(x='Trade Date', y='Cumulative Return', ax=ax1, color='blue')
        ax1.set_title('Cumulative Returns')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return')
        
        # 2. Profit Distribution
        ax2 = plt.subplot(2, 2, 2)
        profit_dist = trades_df[trades_df['Trade Type'] != 'Buy']['Profit Category'].value_counts()
        profit_dist.plot(kind='bar', ax=ax2)
        ax2.set_title('Profit Distribution')
        ax2.set_xlabel('Profit Category')
        ax2.set_ylabel('Trade Count')
        plt.xticks(rotation=45)
        
        # 3. Individual Trade Profits
        ax3 = plt.subplot(2, 2, 3)
        completed_trades = trades_df[trades_df['Trade Type'] != 'Buy']
        ax3.scatter(range(len(completed_trades)), completed_trades['Profit Amount'], 
                    c=completed_trades['Profit Amount'].apply(lambda x: 'g' if x > 0 else 'r'))
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Individual Trade Profits')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Profit Amount')
        
        # 4. Price Chart with Trade Points
        ax4 = plt.subplot(2, 2, 4)
        df['Close'].plot(ax=ax4, label='Close Price', alpha=0.5)
        
        buy_points = trades_df[trades_df['Trade Type'] == 'Buy']
        sell_points = trades_df[trades_df['Trade Type'].isin(['Stop Loss', 'Take Profit'])]
        
        ax4.scatter(buy_points['Trade Date'], buy_points['Trade Price'], 
                    color='g', marker='^', label='Buy')
        ax4.scatter(sell_points['Trade Date'], sell_points['Trade Price'],
                    color='r', marker='v', label='Sell')
        
        ax4.set_title('Price Chart with Trade Points')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存圖表時添加目錄檢查
        import os
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(output_dir, f'trading_results_{timestamp}.png'))
        plt.close()