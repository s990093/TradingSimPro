import pickle
from pyswarm import pso as pyswarm_pso
from alog.pso import pso as alog_pso
from backtest.enhanced_strategy_backtest import EnhancedStrategyBacktest
from config.settings import StrategyConfig
from utils.OptimizationParams import OptimizationParams
from strategies.ml_enhanced_strategy_with_pso import MLEnhancedStrategyPSO
from train import TradeClassifier
from rich.console import Console
from rich.table import Table
import os
import json
import click
from typing import Callable



def get_market_score(total_profit, profit_factor, win_rate, market_type="bear"):
    # 根据市场类型设置目标值
    if market_type == "bull":
        target_total_profit = 100000  # 总利润目标（较高，适应牛市）
        target_profit_factor = 2  # 利润因子目标（更高，追求更好的回报）
        target_win_rate = 0.6  # 胜率目标（适中，适应牛市）
    else:
        target_total_profit = 100  # 总利润目标（较低，适应熊市）
        target_profit_factor = 0.9  # 利润因子目标（较低，控制风险）
        target_win_rate = 0.40  # 胜率目标（较低，适应熊市）

    # 计算与目标值的差异
    total_profit_diff = abs(total_profit - target_total_profit)
    profit_factor_diff = abs(profit_factor - target_profit_factor)
    win_rate_diff = abs(win_rate - target_win_rate)

       # 根据市场类型设定最大差异值
    if market_type == "bull":
        max_total_profit_diff = 200000  # 牛市中允许的最大总利润差异（较大，适应高回报）
        max_profit_factor_diff = 3  # 牛市中允许的最大利润因子差异（较大，追求高回报）
        max_win_rate_diff = 0.4  # 牛市中允许的最大胜率差异（较宽松）
    else:
        max_total_profit_diff = 10000  # 熊市中允许的最大总利润差异（较小，适应较低回报）
        max_profit_factor_diff = 1.5  # 熊市中允许的最大利润因子差异（较小，控制风险）
        max_win_rate_diff = 0.3  # 熊市中允许的最大胜率差异（较小，更注重风险控制）

    # 计算缩放因子，使得最大差异时得分为 0
    alpha_total_profit = 1 / max_total_profit_diff
    alpha_profit_factor = 1 / max_profit_factor_diff
    alpha_win_rate = 1 / max_win_rate_diff

    # 计算每个指标的得分（距离目标越近，得分越高）
    total_profit_score = max(0, 1 - alpha_total_profit * total_profit_diff)
    profit_factor_score = max(0, 1 - alpha_profit_factor * profit_factor_diff)
    win_rate_score = max(0, 1 - alpha_win_rate * win_rate_diff)

    # 市场趋势下的权重调整
    if market_type == "bull":
        weight_total_profit = 0.4  # 牛市中更重视总利润
        weight_profit_factor = 0.3
        weight_win_rate = 0.3
    elif market_type == "bear":
        weight_total_profit = 0.3  # 熊市中控制风险更重要
        weight_profit_factor = 0.4  # 更注重利润因子
        weight_win_rate = 0.3

    # 计算最终得分
    score = (weight_total_profit * total_profit_score) + \
            (weight_profit_factor * profit_factor_score) + \
            (weight_win_rate * win_rate_score)

    return total_profit



def evaluate_strategy(params: list[float]) -> float:

    # Map params to the relevant labels
    params_dict = {}

    # Track indices to dynamically assign params
    param_index = 0

    # Iterate over the labels and assign params based on the order they appear in use_labels
    for label in use_labels:
        # Assign thresholds first
        if label in all_labels_to_threshold:
            params_dict[label] = params[param_index]
            param_index += 1
    
    # Check which position labels should be included based on presence of corresponding thresholds
    if 'large_loss_threshold' in use_labels:
        params_dict['large_loss_position'] = params[-1]
    if 'large_profit_threshold' in use_labels:
        params_dict['large_profit_position'] = params[-3]
    if 'severe_profit_threshold' in use_labels:
        params_dict['severe_profit_position'] = params[-2]
        
    optimization_params = OptimizationParams(
        **params_dict, 

    )
    
    
    ml_strategy.set_threshold(optimization_params)
    ml_strategy.set_stock_data(backtest.data_cache.get_data(show=False))
    backtest.run()
    
    # Assuming you have these initial values already
    total_profit = backtest.get_res()["total_profit"]
    profit_factor = -backtest.get_res()["profit_factor"]
    win_loss_ratio = -backtest.get_res()["win_loss_ratio"]
    
    

    # score = get_market_score(total_profit, profit_factor, win_loss_ratio, "bear")
    return -total_profit
        


def get_pso_function(pso_type: str) -> Callable:
    """根據選擇返回相應的PSO函數"""
    pso_functions = {
        'pyswarm': pyswarm_pso,
        'alog': alog_pso
    }
    return pso_functions.get(pso_type)

@click.command()
@click.option('--maxiter', default=100, help='Maximum number of iterations for PSO')
@click.option('--pso-type', type=click.Choice(['pyswarm', 'alog']), default='pyswarm', 
              help='Type of PSO implementation to use')
def main(maxiter: int, pso_type: str):
    global classifier, ml_strategy, backtest, use_labels, all_labels_to_threshold, included_position_labels
    
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
    

    
    with open(f'res/{StrategyConfig.target_stock}/unique_labels.pkl', 'rb') as file:
            loaded_labels = pickle.load(file)
            
    use_labels = [all_labels_to_threshold[idx] for idx in loaded_labels if 0 <= idx < len(all_labels_to_threshold)]
    # Check if these threshold labels are in use_labels
    threshold_to_position_map = {
        'large_loss_threshold': 'large_loss_position',
        'large_profit_threshold': 'large_profit_position',
        'severe_profit_threshold': 'severe_profit_position'
    }

    # Check which thresholds are in use_labels and create the included_position_labels list accordingly
    included_position_labels = [
        position for threshold, position in threshold_to_position_map.items()
        if threshold in use_labels
    ]


    with open(f"res/{StrategyConfig.target_stock}/best_delta.json", 'r') as f:
        BEST_DELTA = json.load(f)['best_delta']

    classifier = TradeClassifier(model_path=f"models/{StrategyConfig.target_stock}.joblib")

    ml_strategy = MLEnhancedStrategyPSO(
        classifier=classifier,
        delta=BEST_DELTA,       
        max_position=StrategyConfig.MAX_POSITION,     
        stop_loss_pct=StrategyConfig.STOP_LOSS_PCT,
        risk_threshold=0.2,
        cache_dir = f"cache/{StrategyConfig.target_stock}/feature_cache",
        roba_cache = f"cache/{StrategyConfig.target_stock}/predict_proba"
    )

    # Set up the backtest
    backtest = EnhancedStrategyBacktest(
        stock_id=StrategyConfig.target_stock,
        start_date=StrategyConfig.start_date, 
        end_date=StrategyConfig.end_date,
        strategy=ml_strategy,
        show_progress=False
    )
    
    
    # Set the bounds for the PSO optimization
    lower_bounds = [0.0] * len(loaded_labels) + [1] * len(included_position_labels)
    upper_bounds = [1.0] * len(loaded_labels) + [10] * len(included_position_labels)

    # Create a Console instance for rich output
    console = Console()

    # Print the initial parameters
    console.print("Starting PSO optimization with initial parameters:", style="bold green")
    console.print(f"PSO Type: {pso_type}", style="cyan")
    console.print(f"Lower Bounds: {lower_bounds}", style="cyan")
    console.print(f"Upper Bounds: {upper_bounds}", style="cyan")
    console.print(f"Maxiter: {maxiter}", style="cyan")

    # Get the appropriate PSO function
    pso_func = get_pso_function(pso_type)
    if pso_func is None:
        console.print(f"Error: Invalid PSO type '{pso_type}'", style="bold red")
        return

    # Perform PSO to find the optimal parameters
    best_params, best_score = pso_func(evaluate_strategy, lower_bounds, upper_bounds, debug=True, maxiter=maxiter)
 
    # Display the best score
    print(f"Best Score: {-best_score:.4f}")

    # Create a table to display the results
    table = Table(title="Optimal Parameters")

    # Add columns to the table
    table.add_column("Parameter", justify="left", style="cyan")
    table.add_column("Value", justify="right", style="magenta")
    
    # Define parameter names
    param_names = use_labels + included_position_labels

    # Add the best parameters to the table
    for name, value in zip(param_names, best_params):
        table.add_row(name, f"{value:.4f}")

    # Add the best score to the table
    table.add_row("Best Score", f"{-best_score:.4f}")

    # Display the table
    console.print(table)

    # Save the results to a JSON file
    os.makedirs(f"res/{StrategyConfig.target_stock}/optimal_parameters", exist_ok=True)
    score_filename = f'res/{StrategyConfig.target_stock}/optimal_parameters/{-best_score:.4f}.json'
    results_dict = {
        'pso_type': pso_type,
        'parameters': {name: value for name, value in zip(param_names, best_params)},
        'score': float(-best_score)
    }
    with open(score_filename, 'w') as f:
        json.dump(results_dict, f, indent=4)

if __name__ == '__main__':
    main()
