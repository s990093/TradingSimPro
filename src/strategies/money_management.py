import numpy as np
from typing import Dict, List
from rich.console import Console
from rich.table import Table
     
class KellyMoneyManager:
    def __init__(self, 
                 initial_capital: float,
                 win_rate: float = 0.6973,    # 預設勝率
                 win_loss_ratio: float = 1.356,  # 預設獲損比
                 max_risk_pct: float = 0.02,   # 最大風險比例
                 position_scaling: float = 0.5,  # 倉位比例縮放因子
                 max_drawdown_pct: float = 0.1
                 ):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.max_risk_pct = max_risk_pct
        self.position_scaling = position_scaling
        self.trades_history: List[Dict] = []
        self.console = Console()
        self.total_profit = 0  # 初始化 total_profit
        self.position_size = 0  # 初始化 position_size
        self.max_drawdown_pct = max_drawdown_pct
        self.volatility = 0  # 市場波動性
        self.max_loss_limit = initial_capital * 0.2  # 最大可承受虧損
        self.current_drawdown = 0  # 當前回撤
        
    def calculate_kelly_fraction(self) -> float:
        """計算凱利公式建議的資金比例"""
        p = self.win_rate
        q = 1 - p
        b = self.win_loss_ratio
        
        kelly_f = (b * p - q) / b
        
        # 使用縮放因子來降低風險
        kelly_f *= self.position_scaling
        
        # 限制最大風險
        return min(kelly_f, self.max_risk_pct)
        
    def calculate_position_size(self, price: float, stop_loss_pct: float) -> int:
        """計算建議的倉位大小"""
        self.calculate_volatility()  # 更新波動性
        kelly_fraction = self.calculate_kelly_fraction() * (1 - self.volatility)  # 根據波動性調整凱利比例
        
        # 計算每股的風險金額
        risk_per_share = price * stop_loss_pct
        
        # 計算可承受的總風險金額
        total_risk = self.capital * kelly_fraction
        
        # 檢查風險金額是否合理
        if risk_per_share <= 0 or total_risk <= 0:
            return 0  # 返回0以避免無效的股數計算
        
        # 計算建議的股數
        suggested_shares = int(total_risk / risk_per_share)
        
        self.position_size = suggested_shares  # 儲存計算出的 position_size
        return suggested_shares
        
    def update_capital(self, pnl: float):
        """更新資金"""
        self.capital += pnl
        
        self.total_profit += pnl  # 累加 total_profit
        
        self.trades_history.append({
            'capital': self.capital,
            'pnl': pnl,
            'return_pct': (pnl / self.capital) * 100
        })
        
        # 更新當前回撤
        self.current_drawdown = max(0, self.initial_capital - self.capital)
        if self.current_drawdown / self.initial_capital > self.max_drawdown_pct:
            print("回撤超過閾值，暫停交易並重新評估策略。")
            # 這裡可以加入暫停交易的邏輯
        
    def calculate_volatility(self):
        """計算市場波動性"""
        if len(self.trades_history) < 2:
            return
        returns = [trade['pnl'] / self.initial_capital for trade in self.trades_history]
        self.volatility = np.std(returns)
        
    def calculate_sharpe_ratio(self) -> float:
        """計算夏普比率"""
        if len(self.trades_history) < 2:
            return 0.0
        returns = [trade['pnl'] / self.initial_capital for trade in self.trades_history]
        return_mean = np.mean(returns)
        return_std = np.std(returns) if len(returns) > 1 else 1
        risk_free_rate = 0.02
        sharpe = (return_mean - risk_free_rate) / return_std * np.sqrt(250)
        return sharpe
        
    def calculate_sortino_ratio(self) -> float:
        """計算索提諾比率"""
        if len(self.trades_history) < 2:
            return 0.0
        downside_returns = [trade['pnl'] for trade in self.trades_history if trade['pnl'] < 0]
        downside_std = np.std(downside_returns) if downside_returns else 1
        return_mean = np.mean([trade['pnl'] for trade in self.trades_history])
        sortino = (return_mean - 0) / downside_std * np.sqrt(250)
        return sortino
        
    def get_metrics(self) -> Dict:
        """獲取資金管理指標"""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'total_return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
    def _calculate_max_drawdown(self) -> float:
        """計算最大回撤"""
        capitals = [self.initial_capital] + [trade['capital'] for trade in self.trades_history]
        peak = capitals[0]
        max_drawdown = 0
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown * 100
        
    def _calculate_sharpe_ratio(self) -> float:
        """計算夏普比率"""
        if len(self.trades_history) < 2:  # 確保有足夠的樣本數
            return 0.0
            
        returns = [trade['return_pct'] for trade in self.trades_history]
        return_mean = np.mean(returns)
        return_std = np.std(returns) if len(returns) > 1 else 1
        
        # 假設無風險利率為2%
        risk_free_rate = 2.0
        
        # 年化夏普比率 (假設每年約250個交易日)
        sharpe = (return_mean - risk_free_rate) / return_std * np.sqrt(250)
        
        return sharpe

    def monte_carlo_simulation(self, num_simulations: int = 1000, num_trades: int = 100) -> List[float]:
        """使用蒙特卡羅模擬來評估策略的表現"""
        results = []
        
        for _ in range(num_simulations):
            capital = self.initial_capital
            for _ in range(num_trades):
                pnl = self.simulate_trade()  # 假設有一個方法來模擬每次交易的盈虧
                capital += pnl
            results.append(capital)
        
        return results

    def simulate_trade(self) -> float:
        """模擬單次交易的盈虧"""
        # 這裡可以根據策略的勝率和獲損比來計算盈虧
        if np.random.rand() < self.win_rate:
            return self.win_loss_ratio * self.position_size  # 贏的情況
        else:
            return -self.position_size  # 輸的情況

    def show_res(self):
        """Display Kelly criterion results in a formatted table"""

        # Print position size
        self.console.print(f"Calculated position size using Kelly criterion: {self.position_size} shares")
        
        # Create results table
        results_table = Table(title="Backtest Summary")
        results_table.add_column("Metric", style="cyan", justify="right")
        results_table.add_column("Value", style="magenta", justify="right")
        
        results_table.add_row("Total Profit", f"{self.total_profit:.2f} points")
        results_table.add_row("Win Rate", f"{self.win_rate:.2%}")
        results_table.add_row("Win/Loss Ratio", f"{self.win_loss_ratio:.2f}")
        results_table.add_row("Current Capital", f"{self.capital:.2f}")
        results_table.add_row("Total Return (%)", f"{((self.capital - self.initial_capital) / self.initial_capital) * 100:.2f}")
        results_table.add_row("Max Drawdown (%)", f"{self._calculate_max_drawdown():.2f}")
        results_table.add_row("Sharpe Ratio", f"{self._calculate_sharpe_ratio():.2f}")
        results_table.add_row("Sortino Ratio", f"{self.calculate_sortino_ratio():.2f}")

        self.console.print(results_table)