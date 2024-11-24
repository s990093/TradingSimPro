import json
import pickle
from backtest.enhanced_strategy_backtest import EnhancedStrategyBacktest
from config.settings import StrategyConfig
from utils.OptimizationParams import OptimizationParams
from strategies.ml_enhanced_strategy_with_pso import MLEnhancedStrategyPSO
from joblib import Memory

from train import TradeClassifier
from rich.console import Console
import pandas as pd
import click
import os
from rich import print  

all_labels_to_threshold = [
    "severe_loss_threshold",
    "large_loss_threshold",
    "medium_loss_threshold",
    "small_loss_threshold",
    "small_profit_threshold",
    "medium_profit_threshold",
    "large_profit_threshold",
    "severe_profit_threshold"
]
@click.command()
@click.option('--path', required = False, default=None)
def run_enhanced_backtest(path):
    
    if path is None:
        # 查找分数最高的文件
        optimal_params_dir = f"res/{StrategyConfig.target_stock}/optimal_parameters/"
        highest_score_file = None
        highest_score = float('-inf')

        for filename in os.listdir(optimal_params_dir):
            if filename.endswith('.json'):
                score = int(filename.split('.')[0]) 
                if score > highest_score:
                    highest_score = score
                    highest_score_file = filename

        if highest_score_file:
            path = os.path.join(optimal_params_dir, highest_score_file) 
            
            
    print(f"[bold blue]Using parameters from:[/bold blue] {path}")

    with open(path) as f:  
            optimal_params = json.load(f)["parameters"]
            
    with open(f'res/{StrategyConfig.target_stock}/unique_labels.pkl', 'rb') as file:
        loaded_labels = pickle.load(file)
        
        
    use_labels = [all_labels_to_threshold[idx] for idx in loaded_labels if 0 <= idx < len(all_labels_to_threshold)]

    # Assuming optimal_params is a dictionary containing the parameter values
    threshold_params_dict = {}

    # Loop through the use_labels to dynamically build the threshold params
    for label in use_labels:
        # Check if the label exists in optimal_params and add it to the dictionary
        if label in optimal_params:
            threshold_params_dict[label] = optimal_params.get(label)

    # Now, create the OptimizationParams using the dynamically populated dictionary
    threshold_params = OptimizationParams(
        **threshold_params_dict,
        # Assuming the additional params for positions are available in optimal_params
        large_loss_position=optimal_params.get('large_loss_position'),
        large_profit_position=optimal_params.get('large_profit_position'),
        severe_profit_position=optimal_params.get('severe_profit_position')
    )


    # threshold_params.display_params(path=f"res/{StrategyConfig.target_stock}/display_params.png")


    console = Console()
    
    with console.status("[bold green]Running enhanced backtest...") as status:
        # 載入訓練好的模型
        classifier = TradeClassifier(model_path=f"models/{StrategyConfig.target_stock}.joblib")
        
        with open(f"res/{StrategyConfig.target_stock}/best_delta.json", 'r') as f:
            BEST_DELTA = json.load(f)['best_delta']

        # 創建增強版策略
        ml_strategy = MLEnhancedStrategyPSO(
            classifier=classifier,
            delta=BEST_DELTA,       
            max_position=StrategyConfig.MAX_POSITION,     
            stop_loss_pct=StrategyConfig.STOP_LOSS_PCT,
            risk_threshold=0.2,
            threshold_params=threshold_params,
            cache_dir = f"cache/{StrategyConfig.target_stock}/feature_cache",
            roba_cache = f"cache/{StrategyConfig.target_stock}/predict_proba"
        )
        
        
        # 設置回測
        backtest = EnhancedStrategyBacktest(
            stock_id=StrategyConfig.target_stock,
            start_date=StrategyConfig.start_date, 
            end_date=StrategyConfig.end_date,
            strategy=ml_strategy
        )
        
        # 設置股票數據
        ml_strategy.set_stock_data(backtest.data_cache.get_data())
        
        # 執行回測
        backtest.run()
        # backtest.get_kelly_suggestion(StrategyConfig.initial_capital)

        # print(backtest.get_res()["total_profit"])

        # 保存交易記錄
        trades_df = pd.DataFrame(ml_strategy.trades_history)
        trades_df.to_csv(f'res/{StrategyConfig.target_stock}/ml_enhanced_trades.csv', index=False)
        
if __name__ == "__main__":
    run_enhanced_backtest() 