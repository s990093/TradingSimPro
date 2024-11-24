from rich.console import Console
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from rich.table import Table
from rich.console import Console

def classify_profit(profit):
    if profit <= -200:
        return "Heavy Loss"
    elif -200 < profit <= -100:
        return "Large Loss"
    elif -100 < profit <= -50:
        return "Medium Loss"
    elif -50 < profit <= 0:
        return "Small Loss"
    elif 0 < profit <= 50:
        return "Small Profit"
    elif 50 < profit <= 100:
        return "Medium Profit"
    elif 100 < profit <= 200:
        return "Large Profit"
    else:
        return "Heavy Profit"


class TradeType(Enum):
    BUY = "Buy"
    STOP_LOSS = "Stop Loss"
    TAKE_PROFIT = "Take Profit"
    REDUCE = "Reduce"

@dataclass
class TradeRecord:
    trade_date: datetime
    trade_type: TradeType
    trade_price: float
    trigger_price: float
    cumulative_return: float
    profit_category: str
    profit_amount: float
    rsi: float
    volume_ratio: float
    position_size: float = 1.0  
    
    @classmethod
    def create_buy(cls, date, price, cum_return, rsi, vol_ratio, size=1.0):
        return cls(
            trade_date=date,
            trade_type=TradeType.BUY,
            trade_price=price,
            trigger_price=0,
            cumulative_return=cum_return,
            profit_category="Holding",
            profit_amount=0,
            rsi=rsi,
            volume_ratio=vol_ratio,
            position_size=size
        )
    
    @classmethod
    def create_stop_loss(cls, date, price, trigger_price, cum_return, profit, rsi, vol_ratio, size=1.0):
        return cls(
            trade_date=date,
            trade_type=TradeType.STOP_LOSS,
            trade_price=price,
            trigger_price=trigger_price,
            cumulative_return=cum_return,
            profit_category=classify_profit(profit),
            profit_amount=profit,
            rsi=rsi,
            volume_ratio=vol_ratio,
            position_size=size
        )
    
    @classmethod
    def create_take_profit(cls, date, price, cum_return, profit, rsi, vol_ratio, size=1.0):
        return cls(
            trade_date=date,
            trade_type=TradeType.TAKE_PROFIT,
            trade_price=price,
            trigger_price=-1,
            cumulative_return=cum_return,
            profit_category=classify_profit(profit),
            profit_amount=profit,
            rsi=rsi,
            volume_ratio=vol_ratio,
            position_size=size
        )
    
    @classmethod
    def create_reduce(cls, date, price, cum_return, profit, rsi, vol_ratio, reduce_size):
        return cls(
            trade_date=date,
            trade_type=TradeType.REDUCE,
            trade_price=price,
            trigger_price=-1,
            cumulative_return=cum_return,
            profit_category=classify_profit(profit),
            profit_amount=profit,
            rsi=rsi,
            volume_ratio=vol_ratio,
            position_size=reduce_size
        )

    def to_dict(self):
        """Convert TradeRecord to dictionary with standardized column names"""
        return {
            'Trade Date': self.trade_date,
            'Trade Type': self.trade_type.value,  # 使用 .value 來獲取 enum 的字串值
            'Trade Price': self.trade_price,
            'Trigger Price': self.trigger_price,
            'Cumulative Return': self.cumulative_return,
            'Profit Category': self.profit_category,
            'Profit Amount': self.profit_amount,
            'RSI': self.rsi,
            'Volume Ratio': self.volume_ratio,
            'Position Size': self.position_size
        }


@dataclass
class TradingResults:
    total_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown: float
    largest_win: float
    largest_loss: float
    trade_counts: int

    def show_results(self, console: Console = None):
        if console is None:
            console = Console()

        results_table = Table(title="Trading Performance Summary", show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", justify="right", style="green")
        
        # Performance Summary
        results_table.add_row("Total Profit", f"${self.total_profit:.2f}")
        results_table.add_row("Total Trades", f"{self.total_trades} trades")
        results_table.add_row("Winning Trades", f"{self.winning_trades} trades")
        results_table.add_row("Losing Trades", f"{self.losing_trades} trades")
        results_table.add_row("Win Rate", f"{self.win_rate:.2%}")
        
        results_table.add_row("---", "---")
        
        # Profit Metrics
        results_table.add_row("Average Win", f"${self.avg_win:.2f}")
        results_table.add_row("Average Loss", f"${self.avg_loss:.2f}")
        results_table.add_row("Largest Win", f"${self.largest_win:.2f}")
        results_table.add_row("Largest Loss", f"${self.largest_loss:.2f}")
        results_table.add_row("Profit Factor", f"{self.profit_factor:.2f}x")
        
        results_table.add_row("---", "---")
        
        # Risk Metrics
        results_table.add_row("Max Consecutive Wins", f"{self.max_consecutive_wins} trades")
        results_table.add_row("Max Consecutive Losses", f"{self.max_consecutive_losses} trades")
        results_table.add_row("Maximum Drawdown", f"${self.max_drawdown:.2f}")
        
        # Print table with spacing
        console.print("\n")
        console.print(results_table)
        console.print("\n")
