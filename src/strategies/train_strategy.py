import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from datetime import datetime

# from optimizers.strategy_pso_optimizer import OptimizationParams
from strategies.ml_enhanced_strategy import MLEnhancedStrategy
from train import TradeClassifier

def prepare_training_data(historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """準備回測數據和訓練數據"""
    # 假設 historical_data 包含 OHLCV 數據
    train_data = historical_data['2020-01-01':'2022-12-31']
    test_data = historical_data['2023-01-01':]
    return train_data, test_data

def calculate_performance_metrics(returns: pd.Series) -> dict:
    """計算策略績效指標"""
    metrics = {
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
        'win_rate': (returns > 0).mean(),
        'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()),
        'total_return': returns.sum()
    }
    return metrics

def evaluate_strategy(params: OptimizationParams, 
                     strategy: MLEnhancedStrategy,
                     eval_data: pd.DataFrame) -> float:
    """評估策略績效"""
    # 設置策略參數
    strategy.set_risk_weights(
        severe=params.severe_risk_weight,
        large=params.large_risk_weight,
        medium=params.medium_risk_weight
    )
    
    strategy.set_thresholds(
        severe_risk=params.severe_risk_threshold,
        large_risk=params.large_risk_threshold,
        medium_risk=params.medium_risk_threshold,
        high_profit=params.high_profit_threshold,
        medium_profit=params.medium_profit_threshold
    )
    
    # 執行回測
    returns = []
    positions = []
    
    for timestamp, row in eval_data.iterrows():
        action = strategy.update(row['Close'], timestamp)
        returns.append(strategy.current_returns)
        positions.append(strategy.current_position)
    
    returns_series = pd.Series(returns)
    metrics = calculate_performance_metrics(returns_series)
    
    # 返回綜合績效指標（可以根據需求調整權重）
    return (metrics['sharpe_ratio'] * 0.4 + 
            metrics['profit_factor'] * 0.3 + 
            (1 + metrics['win_rate']) * 0.3)

def main():

    # 2. 初始化策略和分類器
    classifier = TradeClassifier()  
    strategy = MLEnhancedStrategy(
        classifier=classifier,
        delta=10.0,
        max_position=5,
        stop_loss_pct=0.02
    )
    
    # # 3. 創建優化器
    # optimizer = StrategyPSOOptimizer(
    #     evaluation_func=lambda params: evaluate_strategy(params, strategy, train_data),
    #     n_particles=30,
    #     dimensions=8,
    #     max_iterations=100
    # )
    
    # # 4. 執行優化
    # print("開始優化策略參數...")
    # best_params, best_performance = optimizer.optimize()
    # print(f"最佳參數：\n{best_params}")
    # print(f"訓練集績效：{best_performance}")
    
    # # 5. 使用最佳參數在測試集上驗證
    # strategy.set_risk_weights(
    #     severe=best_params.severe_risk_weight,
    #     large=best_params.large_risk_weight,
    #     medium=best_params.medium_risk_weight
    # )
    # strategy.set_thresholds(
    #     severe_risk=best_params.severe_risk_threshold,
    #     large_risk=best_params.large_risk_threshold,
    #     medium_risk=best_params.medium_risk_threshold,
    #     high_profit=best_params.high_profit_threshold,
    #     medium_profit=best_params.medium_profit_threshold
    # )
    
    # test_performance = evaluate_strategy(best_params, strategy, test_data)
    # print(f"測試集績效：{test_performance}")
    
    # 6. 儲存最佳參數
    # save_params(best_params, f"strategy_params_{datetime.now().strftime('%Y%m%d')}.json")

if __name__ == "__main__":
    main() 