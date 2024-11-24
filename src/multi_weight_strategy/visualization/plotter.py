import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

class StrategyPlotter:
    @staticmethod
    def plot_strategy_analysis(df_data, best_trades_df, save_path):
        # Calculate Total Return before plotting
        best_trades_df = StrategyPlotter._calculate_total_return(best_trades_df)
        
        plt.figure(figsize=(8, 8))

        # Subplot 1: Trade Actions
        plt.subplot(3, 1, 1)
        StrategyPlotter._plot_trade_actions(df_data, best_trades_df)

        # Subplot 2: Cumulative Profit
        plt.subplot(3, 1, 2)
        StrategyPlotter._plot_cumulative_return(best_trades_df)

        # Subplot 3: Profit Distribution
        plt.subplot(3, 1, 3)
        StrategyPlotter._plot_profit_distribution(best_trades_df)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _plot_trade_actions(df_data, best_trades_df):
        plt.plot(df_data.index, df_data['Close'], label='Close Price', color='blue', alpha=0.5)
        
        # Plot signals
        for action, color, marker in [
            ('Buy', 'green', '^'),
            ('Sell', 'red', 'v'),
            ('Stop', 'orange', 'x')
        ]:
            signals = best_trades_df[best_trades_df['Action'] == action]
            plt.scatter(signals['Date'], signals['Price'], 
                       label=action, marker=marker, color=color, s=13, zorder=5)

        plt.title('Trade Actions on Stock Price', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)

    @staticmethod
    def _plot_cumulative_return(best_trades_df):
        plt.plot(best_trades_df['Date'], best_trades_df['Total Return'], 
                marker='o', linestyle='-', color='purple', markersize=1)
        plt.title('Total Return Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Return', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

    @staticmethod
    def _plot_profit_distribution(best_trades_df):
        console = Console()
        trade_dates = []
        trade_profits = []
        win_trades = []  # Store winning trades
        loss_trades = []  # Store losing trades
        total_loss = 0
        total_profit = 0
        win_count = 0
        
        initial_price = best_trades_df.iloc[0]['Price']
        
        # 使用 Price 列代替 Close 列來計算收益率
        daily_returns = best_trades_df['Price'].pct_change().dropna()
        
        for i in range(1, len(best_trades_df)):
            if (best_trades_df.iloc[i-1]['Action'] == 'Buy' and 
                best_trades_df.iloc[i]['Action'] in ['Sell', 'Stop']):
                profit = best_trades_df.iloc[i]['Price'] - best_trades_df.iloc[i-1]['Price']
                trade_dates.append(best_trades_df.iloc[i]['Date'])
                trade_profits.append(profit)

                # Separate wins and losses
                if profit >= 0:
                    win_trades.append(profit)
                    total_profit += profit
                    win_count += 1
                else:
                    loss_trades.append(profit)
                    total_loss += profit

                console.print(f"[bold]Trade Date:[/bold] {best_trades_df.iloc[i]['Date']}, [bold]Profit:[/bold] {profit:.2f}", 
                             style="green" if profit >= 0 else "red")

        # Calculate various metrics
        total_trades = len(trade_profits)
        win_probability = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit metrics
        total_profit_percentage = (total_profit / initial_price) * 100
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
        
        # Average trade metrics
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Volatility and risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized volatility
        max_drawdown = np.max(np.maximum.accumulate(np.cumsum(trade_profits)) - np.cumsum(trade_profits))
        sharpe_ratio = np.mean(trade_profits) / np.std(trade_profits) if np.std(trade_profits) != 0 else 0
        
        # Calculate expectancy
        expectancy = (win_probability/100 * avg_win) + ((1-win_probability/100) * avg_loss)
        
        # Create performance summary table
        table = Table(title="Trading Performance Summary")
        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="magenta")

        # Profitability metrics
        table.add_row("Total Profit", f"{total_profit:.2f}")
        table.add_row("Total Loss", f"{total_loss:.2f}")
        table.add_row("Total Profit Percentage", f"{total_profit_percentage:.2f}%")
        table.add_row("Profit Factor", f"{profit_factor:.2f}")

        # Trade statistics
        table.add_row("Total Trades", f"{total_trades}")
        table.add_row("Win Probability", f"{win_probability:.2f}%")
        table.add_row("Average Win", f"{avg_win:.2f}")
        table.add_row("Average Loss", f"{avg_loss:.2f}")
        table.add_row("Risk-Reward Ratio", f"{risk_reward_ratio:.2f}")
        
        # Risk metrics
        table.add_row("Annual Volatility", f"{volatility:.2f}%")
        table.add_row("Max Drawdown", f"{max_drawdown:.2f}")
        table.add_row("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        table.add_row("Expectancy", f"{expectancy:.2f}")

        console.print(table)

        # Plot trade distribution
        colors = ['red' if p < 0 else 'green' for p in trade_profits]
        plt.scatter(trade_dates, trade_profits, c=colors, alpha=0.6, s=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.title('Trade Profits Distribution Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Profit', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7) 

    @staticmethod
    def _calculate_total_return(best_trades_df):
        """Calculate cumulative returns for each trade"""
        total_return = 0
        returns = []
        
        for i in range(1, len(best_trades_df)):
            if (best_trades_df.iloc[i-1]['Action'] == 'Buy' and 
                best_trades_df.iloc[i]['Action'] in ['Sell', 'Stop']):
                # Calculate profit for this trade
                profit = best_trades_df.iloc[i]['Price'] - best_trades_df.iloc[i-1]['Price']
                total_return += profit
            returns.append(total_return)
        
        # Add initial 0 return for the first row
        returns.insert(0, 0)
        
        # Add Total Return column
        best_trades_df['Total Return'] = returns
        
        return best_trades_df 