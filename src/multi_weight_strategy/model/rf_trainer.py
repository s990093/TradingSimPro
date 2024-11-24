import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class RFTrainer:
    def __init__(self):
        self.model = None
        
    def add_technical_indicators(self, df_data):
        df_data['RSI'] = talib.RSI(df_data['Close'])
        df_data['MACD'], df_data['MACD_Signal'], _ = talib.MACD(df_data['Close'])
        df_data['SMA_20'] = talib.SMA(df_data['Close'], timeperiod=20)
        df_data['SMA_50'] = talib.SMA(df_data['Close'], timeperiod=50)
        df_data['BB_Upper'], df_data['BB_Middle'], df_data['BB_Lower'] = talib.BBANDS(df_data['Close'])
        return df_data

    def train(self, df_strategy, df_data, signal_columns):
        # 添加技術指標
        df_data = self.add_technical_indicators(df_data)
        
        technical_features = [
            'RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50',
            'BB_Upper', 'BB_Middle', 'BB_Lower'
        ]
        
        X = pd.concat([
            df_strategy[signal_columns],
            df_data[technical_features]
        ], axis=1).fillna(0).values

        future_returns = df_data['Close'].pct_change(5).shift(-5)
        y = (future_returns > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X[:-5], y[:-5], test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        self._print_evaluation(X_test, y_test, signal_columns + technical_features)
        return self.model
    
    def _print_evaluation(self, X_test, y_test, feature_names):
        y_pred = self.model.predict(X_test)
        
        print("\n=== 模型評估報告 ===")
        print("\n分類報告:")
        print(classification_report(y_test, y_pred))
        
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\n具體指標:")
        print(f"準確率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
        print(f"精確率 (Precision): {precision_score(y_test, y_pred):.4f}")
        print(f"召回率 (Recall): {recall_score(y_test, y_pred):.4f}")
        print(f"F1分數: {f1_score(y_test, y_pred):.4f}")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\n特徵重要性排名:")
        print(feature_importance) 