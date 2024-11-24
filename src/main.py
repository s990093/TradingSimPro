from config.settings import StrategyConfig
from get_bset_delta import main as get_bset_delta
from multi_weight_strategy.multi_strategy import MultiWeightStrategy
from multi_weight_strategy.single_strategy import SingleMultiWeightStrategy
from train import main as train_model
from pso_opt import main as pso_optimal
from rich.console import Console
from multi_weight_strategy.train import main as train_multi_weight_model
from multi_weight_strategy.opt_multi import main as opt_multi
import json
# Initialize Rich console
console = Console()

def main():
    opt_multi()
    # train_multi_weight_model() 
    
    # with open(f'res/strategy_parameters.json', 'r') as f:
    #     best_solution = json.load(f)['best_solution']
    
    # strategy = SingleMultiWeightStrategy(
    #     StrategyConfig.target_stock, 
    #     StrategyConfig.start_date, 
    #     StrategyConfig.end_date, 
    #     best_solution, 
    #     f"models/{StrategyConfig.target_stock}_multi_weight_model.joblib"
    # )
    
    
    # best_params = strategy.optimize_parameters()
    # strategy.run() 
    
    # multi_strategy = MultiWeightStrategy(
    #     StrategyConfig.target_stock, 
    #     StrategyConfig.start_date, 
    #     StrategyConfig.end_date, 
    #     best_solution, 
    #     f"models/2498.TW_multi_weight_model.joblib",
    #     max_positions=10,
    #     position_interval_days=100
    # )
    
    
    # # best_params = strategy.optimize_parameters()
    # total_return, trades_df = multi_strategy.run()    
    # print("Optimized Parameters:", best_params)
    
    # console.print("[bold cyan]Step 1: Calculating best delta[/bold cyan]")
    # get_bset_delta()
    # console.print("[bold green]Step 1 completed![/bold green]\n")

    # console.print("[bold cyan]Step 2: Training the model[/bold cyan]")
    # train_model()
    # console.print("[bold green]Step 2 completed![/bold green]\n")

    # console.print("[bold cyan]Step 3: Performing PSO optimization[/bold cyan]")
    # pso_optimal()
    # console.print("[bold green]Step 3 completed![/bold green]\n")

    # console.print("[bold yellow]All steps completed successfully![/bold yellow]")

if __name__ == "__main__":
    main()
