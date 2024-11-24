from typing import Dict, List
import pandas as pd
import numpy as np

class FixedRatioStrategy:
    def __init__(self, 
                 delta: float = 10.0,      # 價格變動增量
                 max_position: int = 5,     # 最大持倉單位
                 stop_loss_pct: float = 0.02,
                 show: bool = True):
        self.delta = delta                  # Δ 價格增量
        self.max_position = max_position    # 最大持倉限制
        self.stop_loss_pct = stop_loss_pct  # 停損比例
        
        # 交易狀態
        self.base_price = 0.0              # 基準價格 P
        self.current_position = 0           # 當前持倉 PZ
        self.entry_price = 0.0             # 進場價格
        self.trades_history: List[Dict] = []
        self.position_cost = 0.0           # 總持倉成本
        self.realized_pnl = 0.0            # 已實現損益
        
        # other
        self.show = show
    def set_delta(self, delta: float):
        self.delta = delta
        
    def reset(self):
        self.base_price = 0.0
        self.current_position = 0
        self.entry_price = 0.0
        self.trades_history = []
        self.position_cost = 0.0
        self.realized_pnl = 0.0
        
    def enter_position(self, price: float, timestamp: pd.Timestamp, position_size: int = 1):
        """建立初始部位
        Args:
            price: 進場價格
            timestamp: 時間戳記
            position_size: 開倉倉位大小，預設為1
        """
        self.base_price = price
        self.entry_price = price
        self.current_position = position_size  # 可以指定初始倉位大小
        self.position_cost = price * position_size  # 初始持倉成本要乘上倉位大小
        
        self.trades_history.append({
            'timestamp': timestamp,
            'action': 'ENTER_LARGE' if position_size > 1 else 'ENTER',
            'price': price,
            'position': self.current_position,
            'level': 0
        })
        
    def update(self, price: float, timestamp: pd.Timestamp) -> str:
        """更新策略狀態"""
        if self.current_position == 0:
            return 'NO_POSITION'
            
        # 計算當前價格相對基準價格的級別
        price_level = int((price - self.base_price) / self.delta)
        
        # 檢查停損條件
        if price <= self.entry_price * (1 - self.stop_loss_pct):
            self._exit_all(price, timestamp, 'STOP_LOSS')
            return 'STOP_LOSS'
        
        # 根據價格級別調整倉位
        target_position = min(max(1, price_level), self.max_position)
        
        if target_position > self.current_position:
            # 增加倉位
            self._add_position(price, timestamp, target_position)
            return 'ADD'
        elif target_position < self.current_position:
            # 減少倉位
            self._reduce_position(price, timestamp, target_position)
            return 'REDUCE'
            
        return 'HOLD'
        
    def _add_position(self, price: float, timestamp: pd.Timestamp, target: int):
        """增加倉位"""
        old_position = self.current_position
        added_units = target - old_position
        self.position_cost += price * added_units  # 更新總持倉成本
        self.current_position = target
        
        self.trades_history.append({
            'timestamp': timestamp,
            'action': 'ADD',
            'price': price,
            'position': self.current_position,
            'level': int((price - self.base_price) / self.delta)
        })
        
    def _reduce_position(self, price: float, timestamp: pd.Timestamp, target: int):
        """減少倉位"""
        old_position = self.current_position
        reduced_units = old_position - target
        
        # 計算已實現損益
        avg_cost = self.position_cost / old_position
        self.realized_pnl += (price - avg_cost) * reduced_units
        
        # 更新持倉成本
        self.position_cost = (self.position_cost / old_position) * target
        self.current_position = target
        
        self.trades_history.append({
            'timestamp': timestamp,
            'action': 'REDUCE',
            'price': price,
            'position': self.current_position,
            'level': int((price - self.base_price) / self.delta)
        })
        
    def _exit_all(self, price: float, timestamp: pd.Timestamp, reason: str):
        """平倉所有部位"""
        if self.current_position > 0:
            avg_cost = self.position_cost / self.current_position
            self.realized_pnl += (price - avg_cost) * self.current_position
            
        self.position_cost = 0
        self.trades_history.append({
            'timestamp': timestamp,
            'action': reason,
            'price': price,
            'position': 0,
            'level': int((price - self.base_price) / self.delta)
        })
        self.current_position = 0
        
    def get_unrealized_pnl(self, current_price: float) -> float:
        """計算未實現損益"""
        if self.current_position == 0:
            return 0.0
        avg_cost = self.position_cost / self.current_position
        return (current_price - avg_cost) * self.current_position
    
    def get_total_pnl(self, current_price: float) -> float:
        """計算總損益（已實現 + 未實現）"""
        return self.realized_pnl + self.get_unrealized_pnl(current_price)