from alog.pso import pso
from config.settings import StrategyConfig
from strategies.dynamic_position_strategy import FixedRatioStrategy
from backtest.strategy_backtest import StrategyBacktest
from rich.console import Console
import json 
import os  

console = Console()

# Initialize strategy and backtest outside the loop
strategy = FixedRatioStrategy(
    delta=0,  
    max_position=StrategyConfig.MAX_POSITION,
    stop_loss_pct=StrategyConfig.STOP_LOSS_PCT,
)

backtest = StrategyBacktest(
    stock_id=StrategyConfig.target_stock,
    start_date=StrategyConfig.start_date,
    end_date=StrategyConfig.end_date,
    strategy=strategy,
    show_progress=False
)

def evaluate_strategy(p):
    backtest.strategy.set_delta(p)
    backtest.run()

    total_profit = backtest.results['total_profit']

    return -total_profit 

def main():
    best_delta = None
    best_result = float('-inf')
    best_distribution = None
    
    # 初始搜索
    for delta in [i * 0.1 for i in range(1, 301)]:
        backtest.strategy.set_delta(delta)
        backtest.run()
        result = -evaluate_strategy(delta)  # 注意這裡使用新的評估函數
        
        if result > best_result:
            best_result = result
            best_delta = delta
    
    print(f"best_delta: {best_delta}", f"best_result: {best_result}")
    # PSO 優化
    lower_bounds = [max(0.1, best_delta - 0.5)]
    upper_bounds = [min(20.0, best_delta + 0.5)]
    maxiter = 100
    
    best_delta, best_result = pso(evaluate_strategy, lower_bounds, upper_bounds, debug=True, maxiter=maxiter)

    # 使用最佳 delta 進行最後一次回測
    backtest.strategy.set_delta(best_delta[0])
    backtest.show_progress = True
    backtest.run()
    final_results = backtest.get_res()
    

    # 儲存結果
    results_data = {
        'best_delta': best_delta[0],
        'total_profit': final_results['total_profit'],
        'trade_distribution': final_results['trade_counts'],
        'metrics': {
            'win_rate': final_results['win_rate'],
            'profit_factor': final_results['profit_factor'],
            'win_loss_ratio': final_results['win_loss_ratio'],
            'max_drawdown': final_results['max_drawdown']
        }
    }
    
    os.makedirs(f"res/{StrategyConfig.target_stock}", exist_ok=True)
    with open(f"res/{StrategyConfig.target_stock}/best_delta.json", 'w') as f:
        json.dump(results_data, f, indent=4)

if __name__ == "__main__":
    main() 