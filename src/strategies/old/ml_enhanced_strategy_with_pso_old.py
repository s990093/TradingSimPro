from typing import Dict, List
import pandas as pd
import numpy as np
import os
import json
from rich.console import Console
from rich.progress import Progress
import matplotlib.pyplot as plt

from train import TradeClassifier
from utils.cache import Cache
from ..dynamic_position_strategy import FixedRatioStrategy
from utils.OptimizationParams import OptimizationParams

console = Console()

class MLEnhancedStrategyPSO(FixedRatioStrategy):
    def __init__(self,
                 classifier: TradeClassifier,
                 delta: float = 10.0,
                 max_position: int = 5,
                 stop_loss_pct: float = 0.02,
                 risk_threshold: float = 0.3,
                 params : OptimizationParams = OptimizationParams(),
                 cache_dir: str = 'cache/feature_cache'): 
        super().__init__(delta, max_position, stop_loss_pct)
        self.classifier = classifier
        self.risk_threshold = risk_threshold
        self.stock_data = None
        self.current_features = None
        self.cache_dir = cache_dir
        self.tech_feature_cache = {}  
        self.combined_feature_cache = {} 
        
        self.roba_cache = Cache("cache/predict_proba")
        
        # pso params
        self.threshold_params = params

        os.makedirs(self.cache_dir, exist_ok=True)  
        
        console.log("Initializing MLEnhancedStrategyPSO...")
        self._load_all_features() 

    def _load_all_features(self):
        """加载所有特征文件到内存"""
        console.log("Loading all feature files into memory...")
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('_combined_features.json'):
                with open(os.path.join(self.cache_dir, filename), 'r') as f:
                    self.combined_feature_cache[filename] = pd.DataFrame(json.load(f))

    def set_stock_data(self, stock_data: pd.DataFrame):
        """設置股票數據用於計算特徵"""
        self.stock_data = stock_data
        
    def _get_current_features(self, price: float, timestamp: pd.Timestamp) -> pd.DataFrame:
        """获取当前时间点的特征"""
        if timestamp not in self.stock_data.index:
            return None
        
        # 直接从内存中查找合并特征
        combined_cache_file = f"{timestamp}_combined_features.json"
        if combined_cache_file in self.combined_feature_cache:
            return self.combined_feature_cache[combined_cache_file]

        # 计算合并特征，因为合并缓存不存在
        cache_file = os.path.join(self.cache_dir, f"{timestamp}.json")
        feature_names = self.classifier.get_orgin_features()

        # 尝试读取技术指标特征缓存
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                tech_features = pd.DataFrame(json.load(f))
        else:
            # 计算技术指标特征
            tech_features = self.classifier.calculate_technical_features(
                self.stock_data.loc[:timestamp]
            ).iloc[-1:]
            feature_names = [name for name in feature_names if name in tech_features.columns]
            tech_features = tech_features[feature_names]
            tech_features.to_json(cache_file)  # 保存技术指标特征到缓存

        # 计算交易相关特征并合并
        combined_features = {
            'price_level': int((price - self.base_price) / self.delta),
            'position_size': self.current_position
        }
        combined_features.update({col: tech_features[col].values[0] for col in tech_features.columns})
        cache_file = os.path.join(self.cache_dir, f"{combined_cache_file}.json")

        # 保存合并特征到缓存并返回
        with open(cache_file, 'w') as f:
            json.dump(combined_features, f)

        return pd.DataFrame([combined_features])
    
    def predict_proba_cache(self, features: pd.DataFrame) -> List[float]:
        """透過快取獲取預測機率"""
        features_tuple = tuple(features.values[0])  # 將特徵轉換為元組以用作字典鍵
        proba = self.roba_cache.get(features_tuple)  # 嘗試從快取中獲取預測
        if proba is None:
            proba = self.classifier.predict_proba(features)[0]
            self.roba_cache.set(features_tuple, proba)  # 將預測結果存儲到快取
        return proba
    
    def update(self, price: float, timestamp: pd.Timestamp) -> str:
        """更新策略狀態"""
        if self.current_position == 0:
            # 考慮開倉
            features = self._get_current_features(price, timestamp)
            if features is not None:
                proba = self.predict_proba_cache(features)
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


            # 如果預測高風險，提前減倉或平倉
            if proba[0] + proba[1] >= self.loss_threshold:
                if self.current_position > 1:
                    self._reduce_position(price, timestamp, self.current_position - 1)
                    return 'REDUCE'
                else:
                    self._exit_all(price, timestamp, 'ML_EXIT')
                    return 'ML_EXIT'
            
            # 如果預測大贏機率高，考慮加倉
            elif proba[6]>= self.threshold_params.large_risk_threshold:  
                if self.current_position < self.max_position:
                    new_position = min(self.current_position + 1, self.max_position)
                    self._add_position(price, timestamp, new_position + 20)
                    return 'ADD'

        # 執行原本的策略邏輯
        return super().update(price, timestamp) 
 
    def set_threshold(self, params: OptimizationParams):
        self.threshold_params = params
