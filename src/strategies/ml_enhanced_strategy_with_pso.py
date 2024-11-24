import hashlib
from typing import Dict, List, Optional
from joblib import Memory
import pandas as pd
import numpy as np
import os
import json
from rich.console import Console
from rich.progress import Progress
import matplotlib.pyplot as plt

from train import TradeClassifier
from utils.cache import Cache
from .dynamic_position_strategy import FixedRatioStrategy
from utils.OptimizationParams import OptimizationParams

console = Console()


# memory = Memory(location="cache", verbose=0)  # 使用 joblib 的 Memory



class MLEnhancedStrategyPSO(FixedRatioStrategy):
    def __init__(self,
                 classifier: TradeClassifier,
                 delta: float = 10.0,
                 max_position: int = 5,
                 stop_loss_pct: float = 0.02,
                 risk_threshold: float = 0.3,
                 threshold_params : OptimizationParams = OptimizationParams(),
                 cache_dir: str = 'cache/feature_cache',
                 roba_cache: str = 'cache/predict_proba'): 
        
        super().__init__(delta, max_position, stop_loss_pct)
        self.classifier = classifier
        self.risk_threshold = risk_threshold
        self.stock_data = None
        self.current_features = None
        self.cache_dir = cache_dir
        self.tech_feature_cache = {}  
        
        self.proba_cache = Cache(roba_cache)
        self.combined_feature_cache = Cache(cache_dir)
        
        self.unique_labels = self.classifier.get_unique_labels()
        # pso params
        self.threshold_params = threshold_params

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

    
    def _get_current_features(self, price: float, timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """获取当前时间点的特征"""
        # 如果时间戳不在股票数据中，返回 None
        if timestamp not in self.stock_data.index:
            return None

        # 尝试从合并特征缓存中获取
        combined_cache_key = f"{timestamp}_combined_features"
        combined_features = self.combined_feature_cache.get(combined_cache_key)
        if combined_features:
            return pd.DataFrame([combined_features])

        # 计算技术指标特征
        tech_cache_key = f"{timestamp}_technical_features"
        tech_features = self.combined_feature_cache.get(tech_cache_key)

        if tech_features is None:
            # 技术特征不存在缓存时，计算并保存
            tech_features_df = self.classifier.calculate_technical_features(
                self.stock_data.loc[:timestamp]
            ).iloc[-1:]

            # 获取所需的特征名称（通过 self.classifier.get_orgin_features()）
            required_feature_names = self.classifier.get_orgin_features()

            # 过滤技术特征，保留需要的特征
            filtered_tech_features = {
                name: value for name, value in tech_features_df.to_dict(orient="records")[0].items()
                if name in required_feature_names
            }

            # 保存技术特征到缓存
            self.combined_feature_cache.set(tech_cache_key, filtered_tech_features)
            tech_features = filtered_tech_features

        # 合并交易相关特征和技术特征
        all_features = {
            'price_level': int((price - self.base_price) / self.delta),
            'position_size': self.current_position,
            **tech_features
        }

        # 过滤特征，确保合并后的特征仅包含所需的特征
        required_feature_names = self.classifier.get_orgin_features()
        filtered_combined_features = {name: value for name, value in all_features.items() if name in required_feature_names}

        # 保存合并特征到缓存
        self.combined_feature_cache.set(combined_cache_key, filtered_combined_features)

        # 返回过滤后的特征
        return pd.DataFrame([filtered_combined_features])
            
    
    def predict_proba_cache(self, features: pd.DataFrame) -> List[float]:
        # 使用 MD5 哈希生成 key
        features_tuple = hashlib.md5(str(tuple(features.values[0])).encode('utf-8')).hexdigest()
        
        # 嘗試從快取中獲取結果
        proba = self.proba_cache.get(features_tuple)
        if proba is None:
            proba = self.classifier.predict_proba(features)[0]
            self.proba_cache.set(features_tuple, proba)
        return proba
    
    def update(self, price: float, timestamp: pd.Timestamp) -> str:
        """更新策略狀態"""
        
        if self.current_position == 0:
            # Consider opening a new position
            features = self._get_current_features(price, timestamp)
            if features is not None:
                proba = self.predict_proba_cache(features)
                
                # If prediction is high risk (based on loss thresholds)
                for i, label in enumerate(self.unique_labels):
                    if label == 'severe_loss_threshold' and proba[i] >= self.threshold_params.severe_loss_threshold:
                        return 'SKIP'
                    elif label == 'large_loss_threshold' and proba[i] >= self.threshold_params.large_loss_threshold:
                        return 'SKIP'
                    elif label == 'medium_loss_threshold' and proba[i] >= self.threshold_params.medium_loss_threshold:
                        return 'SKIP'
                    elif label == 'small_loss_threshold' and proba[i] >= self.threshold_params.small_loss_threshold:
                        return 'SKIP'
                
                # If prediction for large profit is high, consider doubling the position
                for i, label in enumerate(self.unique_labels):
                    if label == 'severe_profit_threshold' and proba[i] >= self.threshold_params.severe_profit_threshold:
                        self.enter_position(price, timestamp, position_size=10)
                        return 'ENTER'
                    elif label == 'large_profit_threshold' and proba[i] >= self.threshold_params.large_profit_threshold:
                        self.enter_position(price, timestamp, position_size=2)
                        return 'ENTER'
                    elif label == 'medium_profit_threshold' and proba[i] >= self.threshold_params.medium_profit_threshold:
                        self.enter_position(price, timestamp, position_size=1.5)
                        return 'ENTER'
                    elif label == 'small_profit_threshold' and proba[i] >= self.threshold_params.small_profit_threshold:
                        self.enter_position(price, timestamp)
                        return 'ENTER'
            
            # Default action to enter a position if no other conditions are met
            self.enter_position(price, timestamp)
            return 'ENTER'
        
        # For the current position, check the conditions for exiting or adjusting
        features = self._get_current_features(price, timestamp)
        if features is not None:
            proba = self.predict_proba_cache(features)
            
            # If prediction is high risk, reduce or exit position
            for i, label in enumerate(self.unique_labels):
                if label == 'severe_loss_threshold' and proba[i] >= self.threshold_params.severe_loss_threshold:
                    self._exit_all(price, timestamp, 'EXIT')
                    return 'EXIT'
                elif label == 'large_loss_threshold' and proba[i] >= self.threshold_params.large_loss_threshold:
                    if self.current_position > 1:
                        self._reduce_position(price, timestamp, self.current_position - self.threshold_params.large_loss_position)
                        return 'REDUCE'
                    else:
                        self._exit_all(price, timestamp, 'EXIT')
                        return 'EXIT'
                elif label == 'medium_loss_threshold' and proba[i] >= self.threshold_params.medium_loss_threshold:
                    if self.current_position > 1:
                        self._reduce_position(price, timestamp, self.current_position - 1)
                        return 'REDUCE'
                elif label == 'small_loss_threshold' and proba[i] >= self.threshold_params.small_loss_threshold:
                    if self.current_position > 1:
                        self._reduce_position(price, timestamp, self.current_position - 1)
                        return 'REDUCE'

            # For profit, consider increasing the position
            for i, label in enumerate(self.unique_labels):
                if label == 'small_profit_threshold' and proba[i] >= self.threshold_params.small_profit_threshold:
                    if self.current_position < self.max_position:
                        new_position = min(self.current_position + 1, self.max_position)
                        self._add_position(price, timestamp, new_position)
                        return 'ADD'
                elif label == 'medium_profit_threshold' and proba[i] >= self.threshold_params.medium_profit_threshold:
                    if self.current_position < self.max_position:
                        new_position = min(self.current_position + 1, self.max_position)
                        self._add_position(price, timestamp, new_position)
                        return 'ADD'
                elif label == 'large_profit_threshold' and proba[i] >= self.threshold_params.large_profit_position:
                    if self.current_position < self.max_position:
                        new_position = min(self.current_position + self.threshold_params.large_profit_position * 3, self.max_position)
                        self._add_position(price, timestamp, new_position)
                        return 'ADD'
                elif label == 'severe_profit_threshold' and proba[i] >= self.threshold_params.severe_profit_threshold:
                    if self.current_position < self.max_position:
                        new_position = min(self.current_position + self.threshold_params.severe_profit_position * 3, self.max_position)
                        self._add_position(price, timestamp, new_position)
                        return 'ADD'

        # If no condition is met, follow the original strategy
        return super().update(price, timestamp)

 
    def set_threshold(self, params: OptimizationParams):
        self.threshold_params = params
