from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Dict, List
from utils.stock_data_cache import StockDataCache
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BaseStrategyBacktest:
    def __init__(self, stock_id: str, start_date: str, end_date: str, strategy: object):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.data_cache = StockDataCache(stock_id, start_date, end_date)
        self.results: Dict = {}
        self.console = Console()
        
        # Create res directory using absolute path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.res_dir = os.path.join(self.base_dir, 'res')
        os.makedirs(self.res_dir, exist_ok=True)

    def calculate_kelly_position(self, price: float, initial_capital: float, win_rate: float, risk_ratio: float = 1.0) -> float:
        kelly_fraction = (win_rate * (risk_ratio + 1) - 1) / risk_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50% of capital
        position_size = (initial_capital * kelly_fraction) / price
        return float(int(position_size))

    def get_kelly_suggestion(self, initial_capital: float) -> Dict:
        if not self.results:
            return {"error": "Please run backtest first to get win rate data"}
        
        df = self.data_cache.get_data()
        current_price = float(df['Close'].iloc[-1].iloc[0])
            
        win_rate = self.results.get('win_rate', 0)
        risk_ratio = self.results.get('win_loss_ratio', 1.0)
        profit_factor = self.results.get('profit_factor', 0)
        total_profit = self.results.get('total_profit', 0)
        
        full_kelly = self.calculate_kelly_position(current_price, initial_capital, win_rate, risk_ratio)
        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25
        
        def calculate_expected_capital(position_size):
            return initial_capital + (position_size * total_profit)
        
        full_kelly_capital = calculate_expected_capital(full_kelly)
        half_kelly_capital = calculate_expected_capital(half_kelly)
        quarter_kelly_capital = calculate_expected_capital(quarter_kelly)
        
        table = Table(title=f"Kelly Criterion Suggestions - {self.stock_id}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Win Rate", f"{win_rate:.2%}")
        table.add_row("Risk Ratio", f"{risk_ratio:.2f}")
        table.add_row("Profit Factor", f"{profit_factor:.2f}")
        table.add_row("Total Profit Points", f"{total_profit:.0f}")
        table.add_row("Initial Capital", f"${initial_capital:,.2f}")
        
        table.add_row("Full Kelly Position", f"{full_kelly:.0f} units (${full_kelly * current_price:,.2f})")
        table.add_row("Half Kelly Position", f"{half_kelly:.0f} units (${half_kelly * current_price:,.2f})")
        table.add_row("Quarter Kelly Position", f"{quarter_kelly:.0f} units (${quarter_kelly * current_price:,.2f})")
        
        table.add_row("Expected Capital (Full)", f"${full_kelly_capital:,.2f} ({((full_kelly_capital/initial_capital)-1)*100:.1f}%)")
        table.add_row("Expected Capital (Half)", f"${half_kelly_capital:,.2f} ({((half_kelly_capital/initial_capital)-1)*100:.1f}%)")
        table.add_row("Expected Capital (Quarter)", f"${quarter_kelly_capital:,.2f} ({((quarter_kelly_capital/initial_capital)-1)*100:.1f}%)")
        
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]Kelly Criterion Analysis[/bold green]",
            title="Position Sizing Suggestions",
            border_style="green"
        ))
        self.console.print(table)
        
        self.plot_kelly_analysis(initial_capital)
        
        return {
            "full_kelly": {
                "position_size": full_kelly,
                "capital_fraction": full_kelly * current_price / initial_capital,
                "expected_capital": full_kelly_capital
            },
            "half_kelly": {
                "position_size": half_kelly,
                "capital_fraction": half_kelly * current_price / initial_capital,
                "expected_capital": half_kelly_capital
            },
            "quarter_kelly": {
                "position_size": quarter_kelly,
                "capital_fraction": quarter_kelly * current_price / initial_capital,
                "expected_capital": quarter_kelly_capital
            },
            "metrics_used": {
                "win_rate": win_rate,
                "risk_ratio": risk_ratio,
                "profit_factor": profit_factor,
                "total_profit": total_profit
            }
        } 
        
    def _show_res(self):
         # 使用 Rich table 來展示結果
        table = Table(title=f"Backtest Results - {self.stock_id}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Total Profit", f"{self.results['total_profit']:.0f} points")
        table.add_row("Total Trades", str(self.results['total_trades']))
        table.add_row("Win Rate", f"{self.results['win_rate']:.2%}")
        table.add_row("Profit Factor", f"{self.results['profit_factor']:.3f}")
        table.add_row("Win/Loss Ratio", f"{self.results['win_loss_ratio']:.3f}")
        table.add_row("Max Drawdown", f"{self.results['max_drawdown']:.0f}")
        
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold green]Backtest Completed![/bold green]",
            title="Status",
            border_style="green"
        ))
        self.console.print(table)

        # Calculate trade counts
        trade_counts = self.results['trade_counts']
        total_trade_counts = sum(trade_counts)  # Calculate total trades for percentage

        labels = [
            'Extreme Loss (<-200)',  # 极大亏损
            'Large Loss (-200~-100)',  # 重大亏损
            'Medium Loss (-100~-50)',  # 大额亏损
            'Small Loss (-50~0)',  # 中额亏损
            'Tiny Loss (0~50)',  # 微亏损
            'Small Profit (50~100)',  # 小额获利
            'Medium Profit (100~200)',  # 中额获利
            'Large Profit (>200)',  # 大额获利
        ]

        weights_table = Table(title="Class Trade Counts", show_header=True, header_style="bold blue")
        weights_table.add_column("Label", style="cyan")
        weights_table.add_column("Trade Count", justify="right", style="green")
        weights_table.add_column("Percentage", justify="right", style="yellow") 

        # Add rows for trade counts with percentages
        for i in range(len(labels)):
            label = labels[i]
            count = trade_counts[i]
            percentage = (count / total_trade_counts * 100) if total_trade_counts > 0 else 0
            weights_table.add_row(label, str(count), f"{percentage:.1f}%")  # Add trade count and percentage

        # Display table
        self.console.print(weights_table)
        
    def plot_kelly_analysis(self, initial_capital: float):
        if not self.results:
            print("Please run backtest first to get results")
            return
        
        df = self.data_cache.get_data()
        current_price = float(df['Close'].iloc[-1].iloc[0])
        
        win_rate = self.results.get('win_rate', 0)
        risk_ratio = self.results.get('win_loss_ratio', 1.0)
        total_profit = self.results.get('total_profit', 0)
        
        full_kelly = self.calculate_kelly_position(current_price, initial_capital, win_rate, risk_ratio)
        half_kelly = full_kelly * 0.5
        quarter_kelly = full_kelly * 0.25
        
        positions = [quarter_kelly, half_kelly, full_kelly]
        position_labels = ['Quarter Kelly', 'Half Kelly', 'Full Kelly']
        
        expected_capitals = [initial_capital + (pos * total_profit) for pos in positions]
        capital_changes = [((cap / initial_capital) - 1) * 100 for cap in expected_capitals]
        capital_requirements = [pos * current_price for pos in positions]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Recommended Position Sizes', 'Capital Requirements', 'Expected Return (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=position_labels, y=positions, 
                   text=[f'{int(pos):,} shares' for pos in positions],
                   textposition='auto',
                   marker_color=['lightgreen', 'green', 'darkgreen'],
                   name='Position Size'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=position_labels, y=capital_requirements,
                   text=[f'${int(cap):,} ({int(cap/1000)}K)' for cap in capital_requirements],
                   textposition='auto',
                   marker_color=['lightblue', 'blue', 'darkblue'],
                   name='Capital Required'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=position_labels, y=capital_changes,
                   text=[f'{change:.1f}%' for change in capital_changes],
                   textposition='auto',
                   marker_color=['pink', 'red', 'darkred'],
                   name='Expected Return'),
            row=3, col=1
        )
        
        fig.update_layout(
            title_text=f'Kelly Criterion Analysis - {self.stock_id} (Initial Capital: ${initial_capital:,.0f})',
            height=900,
            showlegend=False,
            title_x=0.5,
        )
        
        metrics_text = (
            f'Win Rate: {win_rate:.1%}<br>'
            f'Risk Ratio: {risk_ratio:.2f}<br>'
            f'Profit Factor: {self.results["profit_factor"]:.2f}<br>'
            f'Initial Capital: ${initial_capital:,.2f}<br>'
            f'Total Profit Points: {total_profit:.0f}'
        )
        
        fig.add_annotation(
            x=0,
            y=-0.2,
            xref='paper',
            yref='paper',
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            align='left',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1,
            borderpad=4
        )
        
        save_path = os.path.join(self.res_dir, f'kelly_analysis_{self.strategy.__class__.__name__}.html')
        fig.write_html(save_path)
        
        save_path_png = os.path.join(self.res_dir, f'kelly_analysis_{self.strategy.__class__.__name__}.png')
        fig.write_image(save_path_png)
        
        fig.show()

    def plot_class_weights_and_counts(self):
        # Prepare data for trade counts
        trade_counts = self.results['trade_counts']
        
        # Calculate proportions
        total_trades = sum(trade_counts)
        proportions = [count / total_trades * 100 if total_trades > 0 else 0 for count in trade_counts]

        # Directly specify colors for each class label
        colors = [
            plt.cm.Blues(0.8),  # Large Loss (<-100) - Dark Blue
            plt.cm.Blues(0.6),  # Small Loss (-100~-50) - Medium Blue
            plt.cm.Blues(0.4),  # Tiny Loss (-50~0) - Light Blue
            plt.cm.Reds(0.4),   # Tiny Profit (0~50) - Light Red
            plt.cm.Reds(0.6),   # Small Profit (50~100) - Medium Red
            plt.cm.Reds(0.8),   # Large Profit (>100) - Dark Red
        ]

        # Create a bar plot for trade counts
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(trade_counts)), trade_counts, color=colors, edgecolor='black')  # Use range instead of class_weights.keys()
        
        # Add titles and labels
        plt.title('Trade Counts', fontsize=16)
        plt.xlabel('Class Labels', fontsize=14)
        plt.ylabel('Trade Count', fontsize=14)
        plt.xticks(ticks=range(len(trade_counts)), labels=[
            'Large Loss (<-100)', 'Small Loss (-100~-50)', 'Tiny Loss (-50~0)',
            'Tiny Profit (0~50)', 'Small Profit (50~100)', 'Large Profit (>100)',
        ], rotation=45)

        # Add proportions on top of the bars
        for bar, proportion in zip(bars, proportions):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval} ({proportion:.1f}%)', 
                    ha='center', va='bottom', fontsize=10, color='black')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
        
        
               
    def _calculate_trade_counts(self, trades):
        trade_counts = [0] * 8  # 新增8个分类
        for trade in trades:
            if trade < -200:
                trade_counts[0] += 1  # 极大亏损
            elif -200 <= trade < -100:
                trade_counts[1] += 1  # 重大亏损
            elif -100 <= trade < -50:
                trade_counts[2] += 1  # 大额亏损
            elif -50 <= trade < 0:
                trade_counts[3] += 1  # 中额亏损
            elif 0 <= trade < 50:
                trade_counts[4] += 1  # 微亏损
            elif 50 <= trade < 100:
                trade_counts[5] += 1  # 小额获利
            elif 100 <= trade < 200:
                trade_counts[6] += 1  # 中额获利
            elif trade >= 200:
                trade_counts[7] += 1  # 大额获利

        
        # 確保統計數量中性
        total_trades = sum(trade_counts)
        assert total_trades == len(trades), f"Total trades counted {total_trades} does not match input trades {len(trades)}"
        
        return trade_counts


    def get_res(self):
            return self.results
        
    def display_charts(self, times, positions, stock_prices, buy_points, risk_reward):
        # Create a figure for all plots
        
        plt.figure(figsize=(10, 8))

        # Plot for positions
        plt.subplot(4, 1, 1)  # 6 rows, 1 column, 1st subplot
        plt.plot(times, positions, label='Positions', color='blue', linewidth=1)
        plt.title('Position Changes Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Positions', fontsize=14)
        plt.legend()
        plt.grid()

        # Plot for stock prices and buy points
        plt.subplot(4, 1, 2)  # 6 rows, 1 column, 2nd subplot
        plt.plot(times, stock_prices, label='Stock Prices', color='green', linewidth=2)
        plt.title(f'{self.stock_id} Stock Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Stock Price', fontsize=14)
        plt.legend()
        
        # Mark buy points on the stock price chart
        for buy_time, buy_price in buy_points:
            plt.plot(buy_time, buy_price, marker='v', color='lime', markersize=5, label='Buy Point')

        plt.grid()
        
        # Plot for total points over time
        
        plt.subplot(4, 1, 3)  # 6 rows, 1 column, 4th subplot
        current = 0.0
        cumulative_pnls = []  # To store cumulative P&L at each trade

        # Calculate cumulative P&L for each trade
        for pnl in self.results['trades']:
            current += pnl
            cumulative_pnls.append(current)

        # Plot the cumulative P&L over time
        plt.plot(self.results['trade_times'], cumulative_pnls, marker='o', linestyle='-', color='royalblue', markersize=1)
        plt.title('Cumulative Trade P&L Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Cumulative Trade P&L', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels for better readability
        plt.yticks(fontsize=12)  # Adjust y-axis label font size
        plt.grid(color='gray', linestyle='--', linewidth=0.5)  # Add grid with custom style            
        
        # Prepare data for trade counts
        trade_counts = self.results['trade_counts']
        
        # Calculate proportions
        total_trades = sum(trade_counts)
        proportions = [
            count / total_trades * 100 if total_trades > 0 else 0 
            for count in trade_counts
        ]

        # Directly specify colors for each class label
        colors = [
            plt.cm.Blues(0.8),  # Extreme Loss (<-150) - Dark Blue
            plt.cm.Blues(0.6),  # Large Loss (-150~-100) - Medium Blue
            plt.cm.Blues(0.4),  # Medium Loss (-100~-50) - Light Blue
            plt.cm.Reds(0.4),   # Small Loss (-50~0) - Light Red
            plt.cm.Reds(0.6),   # Tiny Loss (0~50) - Medium Red
            plt.cm.Reds(0.8),   # Small Profit (50~100) - Dark Red
            plt.cm.Greens(0.6),  # Medium Profit (100~200) - Medium Green
            plt.cm.Greens(0.8)   # Large Profit (>200) - Dark Green
        ]

        # Create a bar plot for trade counts
        plt.subplot(4, 1, 4)  # 6 rows, 1 column, 5th subplot
        bars = plt.bar(range(len(trade_counts)), trade_counts, color=colors, edgecolor='black')
        
        # Add titles and labels
        plt.title('Trade Counts', fontsize=16)
        plt.xlabel('Class Labels', fontsize=14)
        plt.ylabel('Trade Count', fontsize=14)
        # Set x-ticks with the correct category labels and rotate them for better readability
        plt.xticks(
            ticks=range(len(trade_counts)),
            labels=[
                'Extreme Loss (<-200)', 'Large Loss (-200~-100)', 'Medium Loss (-100~-50)', 
                'Small Loss (-50~0)', 'Tiny Loss (0~50)', 'Small Profit (50~100)', 
                'Medium Profit (100~200)', 'Large Profit (>200)'
            ], 
            rotation=45
        )
        # Add proportions on top of the bars
        for bar, proportion in zip(bars, proportions):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval} ({proportion:.1f}%)', 
                     ha='center', va='bottom', fontsize=10, color='black')

        plt.tight_layout() 
        save_dir = os.path.join(self.res_dir, f'{self.stock_id}')
        os.makedirs(save_dir, exist_ok=True)

        # 保存圖表
        save_path = os.path.join(save_dir, f'{self.stock_id}_charts.png')
        plt.savefig(save_path) 
        plt.show()

