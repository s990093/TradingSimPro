from typing import Dict, List
import pandas as pd
import numpy as np
import os
import pickle

from train import TradeClassifier
from .dynamic_position_strategy import FixedRatioStrategy
from utils.cache import Cache

class MLEnhancedStrategy(FixedRatioStrategy):
    def __init__(self, 
                 classifier: TradeClassifier,
                 delta: float = 10.0,
                 max_position: int = 5,
                 stop_loss_pct: float = 0.02,
                 risk_threshold: float = 0.3,
                 initial_capital: float = 1000000,
                 cache_dir: str = 'cache/predict_proba'):  
        super().__init__(delta, max_position, stop_loss_pct)
        
        self.classifier = classifier
        self.risk_threshold = risk_threshold
        self.stock_data = None
        self.current_features = None
        self.cache = Cache(cache_dir)
        
    def set_stock_data(self, stock_data: pd.DataFrame):
        """設置股票數據用於計算特徵"""
        self.stock_data = stock_data
        
    def _get_current_features(self, price: float, timestamp: pd.Timestamp) -> pd.DataFrame:
        """獲取當前時間點的特徵"""
        if timestamp not in self.stock_data.index:
            return None
            
        # 計算技術指標特徵
        tech_features = self.classifier.calculate_technical_features(
            self.stock_data.loc[:timestamp]
        ).iloc[-1:]
        
        # 添加交易相關特徵
        features = pd.DataFrame(index=[0])
        features['price_level'] = int((price - self.base_price) / self.delta)
        features['position_size'] = self.current_position
        
        # 合併特徵
        for col in tech_features.columns:
            features[col] = tech_features[col].values[0]
            
        return features
        
    def predict_proba_cache(self, features: pd.DataFrame) -> List[float]:
        """透過快取獲取預測機率"""
        features_tuple = tuple(features.values[0])  # 將特徵轉換為元組以用作字典鍵
        proba = self.cache.get(features_tuple)  # 嘗試從快取中獲取預測
        if proba is None:
            proba = self.classifier.predict_proba(features)[0]
            self.cache.set(features_tuple, proba)  # 將預測結果存儲到快取
        return proba

    def update(self, price: float, timestamp: pd.Timestamp) -> str:
        """更新策略狀態"""
        if self.current_position == 0:
            # 考慮開倉
            features = self._get_current_features(price, timestamp)
            if features is not None:
                proba = self.predict_proba_cache(features)  # 使用快取獲取預測
                # 如果預測高風險（重大虧損和大額虧損的機率較高），則不開倉
                if proba[0] + proba[1] >= self.risk_threshold:  # 0:重大虧損 + 1:大額虧損
                    return 'ML_SKIP'
                # 如果預測重大獲利的機率高，則加倍開倉
                elif proba[7] >= 0.6:  # 7:重大獲利 (profit > 200) 的機率超過 60%
                    self.enter_position(price, timestamp, position_size=3)  # 開倉兩倍
                    return 'ENTER_DOUBLE'
            self.enter_position(price, timestamp)
            return 'ENTER'
        
        # 獲取當前特徵和預測
        features = self._get_current_features(price, timestamp)
        if features is not None:
            proba = self.predict_proba_cache(features)  

            if proba[0] + proba[1] >= self.risk_threshold:
                # 增加對高風險的反應，提前減倉
                if self.current_position > 1:
                    self._reduce_position(price, timestamp, self.current_position - 1)
                    return 'ML_REDUCE'
                else:
                    self._exit_all(price, timestamp, 'ML_EXIT')
                    return 'ML_EXIT'
            
            # 如果預測大贏機率高，考慮加倉
            elif proba[6] >= self.risk_threshold:  
                if self.current_position < self.max_position:
                    new_position = min(self.current_position + 1, self.max_position)
                    self._add_position(price, timestamp, new_position + 20)
                    return 'ML_ADD'
                

        # 執行原本的策略邏輯
        return super().update(price, timestamp) 

    # def set_risk_weights(self, severe: float, large: float, medium: float):
    #     self.severe_risk_weight = severe
    #     self.large_risk_weight = large
    #     self.medium_risk_weight = medium

    # def set_thresholds(self, 
    #                    severe_risk: float,
    #                    large_risk: float,
    #                    medium_risk: float,
    #                    high_profit: float,
    #                    medium_profit: float):
    #     self.severe_risk_threshold = severe_risk
    #     self.large_risk_threshold = large_risk
    #     self.medium_risk_threshold = medium_risk
    #     self.high_profit_threshold = high_profit
    #     self.medium_profit_threshold = medium_profit


