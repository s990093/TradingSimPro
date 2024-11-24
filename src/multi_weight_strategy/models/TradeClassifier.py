import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Tuple, Dict
import joblib
from rich.console import Console
from rich.table import Table
from backtest.strategy_backtest import StrategyBacktest
from config.settings import StrategyConfig
from strategies.dynamic_position_strategy import FixedRatioStrategy
import talib as ta
import os
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class TradeClassifier:
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self.console = Console()
        self.feature_names = None
        self.tech_features_cache = None
        os.makedirs("models", exist_ok=True)
        self.categories = [
            'Major Loss', 'Large Loss', 'Medium Loss', 'Small Loss',
            'Small Profit', 'Medium Profit', 'Large Profit', 'Major Profit'
        ]
        
        # 如果模型文件存在，在初始化時就載入
        if os.path.exists(self.model_path):
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.feature_names = saved_data['feature_names']
            self.labels = saved_data['labels']

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標特徵，如果已有快取則直接返回"""
        if self.tech_features_cache is not None and self.tech_features_cache.index.equals(df.index):
            return self.tech_features_cache
        
        features = pd.DataFrame(index=df.index)
        
        # Convert to numpy arrays and ensure they're float64 and 1-dimensional
        close_arr = df['Close'].astype(float).values.flatten()
        high_arr = df['High'].astype(float).values.flatten()
        low_arr = df['Low'].astype(float).values.flatten()
        volume_arr = df['Volume'].astype(float).values.flatten()
        open_arr = df['Open'].astype(float).values.flatten()
        
        # Verify array shapes
        if close_arr.ndim != 1:
            close_arr = close_arr.flatten()
        if high_arr.ndim != 1:
            high_arr = high_arr.flatten()
        if low_arr.ndim != 1:
            low_arr = low_arr.flatten()
        if volume_arr.ndim != 1:
            volume_arr = volume_arr.flatten()
        if open_arr.ndim != 1:
            open_arr = open_arr.flatten()
        
        # 基礎價格特徵
        features['Price_SMA5_Dist'] = (close_arr - ta.SMA(close_arr, timeperiod=5)) / ta.SMA(close_arr, timeperiod=5)
        features['Price_SMA10_Dist'] = (close_arr - ta.SMA(close_arr, timeperiod=10)) / ta.SMA(close_arr, timeperiod=10)
        features['Price_SMA20_Dist'] = (close_arr - ta.SMA(close_arr, timeperiod=20)) / ta.SMA(close_arr, timeperiod=20)
        features['Price_SMA60_Dist'] = (close_arr - ta.SMA(close_arr, timeperiod=60)) / ta.SMA(close_arr, timeperiod=60)
        
        # 移動平均線交叉信號
        features['SMA5_10_Cross'] = ta.SMA(close_arr, timeperiod=5) - ta.SMA(close_arr, timeperiod=10)
        features['SMA10_20_Cross'] = ta.SMA(close_arr, timeperiod=10) - ta.SMA(close_arr, timeperiod=20)
        features['SMA20_60_Cross'] = ta.SMA(close_arr, timeperiod=20) - ta.SMA(close_arr, timeperiod=60)
        
        # 動量指標
        features['RSI'] = ta.RSI(close_arr, timeperiod=14)
        features['RSI_5'] = ta.RSI(close_arr, timeperiod=5)
        features['RSI_21'] = ta.RSI(close_arr, timeperiod=21)
        features['MACD'], features['MACD_Signal'], features['MACD_Hist'] = ta.MACD(close_arr)
        features['MOM'] = ta.MOM(close_arr, timeperiod=10)
        features['ROC'] = ta.ROC(close_arr, timeperiod=10)
        
        # 布林帶指標
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close_arr, timeperiod=20)
        features['BB_Position'] = (close_arr - bb_lower) / (bb_upper - bb_lower)
        features['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # KD隨機指標
        features['STOCH_K'], features['STOCH_D'] = ta.STOCH(high_arr, low_arr, close_arr)
        features['STOCH_RSI'] = ta.STOCHRSI(close_arr)[0]
        
        # 趨勢指標
        features['ADX'] = ta.ADX(high_arr, low_arr, close_arr)
        features['DI_PLUS'] = ta.PLUS_DI(high_arr, low_arr, close_arr)
        features['DI_MINUS'] = ta.MINUS_DI(high_arr, low_arr, close_arr)
        features['AROON_UP'], features['AROON_DOWN'] = ta.AROON(high_arr, low_arr)
        
        # 成交量指標
        features['Volume_Ratio'] = volume_arr / ta.SMA(volume_arr, timeperiod=5)
        features['OBV'] = ta.OBV(close_arr, volume_arr)
        features['AD'] = ta.AD(high_arr, low_arr, close_arr, volume_arr)
        features['CMF'] = ta.ADOSC(high_arr, low_arr, close_arr, volume_arr)
        
        # 波動率指標
        features['ATR'] = ta.ATR(high_arr, low_arr, close_arr)
        features['NATR'] = ta.NATR(high_arr, low_arr, close_arr)
        features['Volatility'] = ta.STDDEV(close_arr, timeperiod=20)
        
        # 價格形態指標
        features['DOJI'] = ta.CDLDOJI(open_arr, high_arr, low_arr, close_arr)
        features['HAMMER'] = ta.CDLHAMMER(open_arr, high_arr, low_arr, close_arr)
        features['SHOOTING_STAR'] = ta.CDLSHOOTINGSTAR(open_arr, high_arr, low_arr, close_arr)
        
        # 自定義組合特徵
        features['RSI_SMA5_Ratio'] = features['RSI'] / features['Price_SMA5_Dist']
        features['RSI_SMA5_Ratio'] = features['RSI_SMA5_Ratio'].clip(-100, 100)

        features['Volume_Price_Ratio'] = features['Volume_Ratio'] * features['Price_SMA5_Dist']
        features['BB_RSI_Signal'] = features['BB_Position'] * features['RSI']
        features['Trend_Strength'] = features['ADX'] * (features['DI_PLUS'] - features['DI_MINUS'])
        features['Volume_Trend'] = features['Volume_Ratio'] * features['Price_SMA5_Dist']
        features['Momentum_Signal'] = features['RSI'] * features['MACD_Hist']
        
        # 價格變化率
        for period in [1, 3, 5, 10, 20]:
            features[f'Price_Change_{period}'] = ta.ROC(close_arr, timeperiod=period)
        
        # 波動率變化
        for period in [5, 10, 20]:
            features[f'Volatility_{period}'] = ta.STDDEV(close_arr, timeperiod=period)
        
        # 儲存結果到快取
        self.tech_features_cache = features
        
        return features.fillna(0)
        
    def prepare_data(self, trades_df: pd.DataFrame, stock_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """準備訓練數據"""
        tech_features = self.calculate_technical_features(stock_data)
        
        # 合併交易記錄與技術指標
        features = pd.DataFrame()
        
        # 交易相關特徵
        features['price_level'] = trades_df['level']
        features['position_size'] = trades_df['position']
        
        # 添加技術指標特徵
        for idx, row in trades_df.iterrows():
            timestamp = row['timestamp']
            if timestamp in tech_features.index:
                for col in tech_features.columns:
                    features.at[idx, col] = tech_features.loc[timestamp, col]
        
        # 分類標籤：將獲利情況分類得更細緻
        def classify_profit(profit):
            if profit <= -200:
                return 0  # 重大虧損
            elif -200 < profit <= -100:
                return 1  # 大額虧損
            elif -100 < profit <= -50:
                return 2  # 中額虧損
            elif -50 < profit <= 0:
                return 3  # 小額虧損
            elif 0 < profit <= 50:
                return 4  # 小額獲利
            elif 50 < profit <= 100:
                return 5  # 中額獲利
            elif 100 < profit <= 200:
                return 6  # 大額獲利
            else:
                return 7  # 重大獲利
        
        # 計算每筆交易的獲利
        profits = []
        for i in range(0, len(trades_df)-1, 2):
            if i+1 >= len(trades_df):
                break
            entry = trades_df.iloc[i]
            exit = trades_df.iloc[i+1]
            profit = (exit['price'] - entry['price']) * entry['position']
            profits.extend([profit, profit])
        
        # Add this check to ensure profits list matches features length
        if len(profits) < len(features):
            # Handle the last unpaired trade by duplicating the last profit
            profits.append(profits[-1] if profits else 0)
        elif len(profits) > len(features):
            # Trim excess profits if necessary
            profits = profits[:len(features)]
        
        labels = pd.Series(profits).apply(classify_profit)
                
        # 特徵工程：添加特徵組合（添加安全檢查以避免除以零）
        features['RSI_SMA5_Ratio'] = np.where(
            features['Price_SMA5_Dist'] != 0,
            features['RSI'] / features['Price_SMA5_Dist'],
            0
        )

        features['Volume_Price_Ratio'] = features['Volume_Ratio'] * features['Price_SMA5_Dist']
        features['BB_RSI_Signal'] = features['BB_Position'] * features['RSI']
        
        # 替換無限值為0
        features = features.replace([np.inf, -np.inf], 0)
        
        # 特徵選擇
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        selector.fit(features, labels)
        features_selected = features.loc[:, selector.get_support()]
        
        return features_selected, labels
        
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """訓練模型"""
        # 保存特徵名稱
        self.feature_names = features.columns.tolist()
        
        # 分割訓練和測試數據
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # 調整類別權重以更好地處理不平衡問題
        class_weights = {
            0: 2.5,  # 重大虧損 (提高權重)
            1: 2.0,  # 大額虧損
            2: 1.8,  # 中額虧損
            3: 1.5,  # 小額虧損
            4: 1.2,  # 小額獲利
            5: 1.5,  # 中額獲利
            6: 2.0,  # 大額獲利
            7: 2.5   # 重大獲利 (提高權重)
        }
        
        # 定義參數網格
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [15, 20, 25],
            'min_samples_split': [4, 5, 6],
            'min_samples_leaf': [1, 2]
        }
        
        # 使用網格搜索
        grid_search = GridSearchCV(
            RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(features, labels)
        self.model = grid_search.best_estimator_
        
        # 評估模型
        y_pred = self.model.predict(X_test)
        
        # 顯示評估結果
        unique_labels = list(set(y_test))
        # Save data to a pickle file
        with open(f'res/{StrategyConfig.target_stock}/unique_labels.pkl', 'wb') as file:
            pickle.dump(unique_labels, file)
                
        
        self._display_metrics(y_test, y_pred)
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_feature_importance()
        
        # 保存模型和特徵名稱
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'labels': labels,
            'model_parameters': self.model.get_params()
        }, self.model_path)
        
    def _display_metrics(self, y_true, y_pred):
        """顯示模型評估指標"""
        
        unique_labels = list(set(y_true))
        filtered_categories = [self.categories[i] for i in unique_labels]
        report = classification_report(y_true, y_pred, target_names=filtered_categories)
        self.console.print("\n[bold]Classification Report:[/bold]")
        self.console.print(report)

        # 添加 F1 Score 顯示
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.console.print(f"\n[bold]Weighted F1 Score:[/bold] {f1:.3f}")
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        """繪製混淆矩陣"""
        categories = [
            'Major Loss', 'Large Loss', 'Medium Loss', 'Small Loss',
            'Small Profit', 'Medium Profit', 'Large Profit', 'Major Profit'
        ]
        
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories,
                   yticklabels=categories)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'res/{StrategyConfig.target_stock}/confusion_matrix.png')
        plt.close()
        
    def _plot_feature_importance(self):
        """繪製特徵重要性圖表"""
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        # 排序特徵重要性
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'res/{StrategyConfig.target_stock}/feature_importance.png')
        plt.close()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """預測交易結果"""
        if self.model is None:
            # 載入模型和特徵名稱
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.feature_names = saved_data['feature_names']
        
        # 確保輸入特徵與訓練時的特徵相同
        features = features[self.feature_names]
        return self.model.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """預測交易結果的機率分布"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first or provide a valid model path.")
        
        features = features[self.feature_names]
        
        
        return self.model.predict_proba(features)
    
    def get_orgin_features(self):
        return self.feature_names
    
    def get_unique_labels(self):
        with open(f'res/{StrategyConfig.target_stock}/unique_labels.pkl', 'rb') as file:
            loaded_labels = pickle.load(file)
            
        return loaded_labels


