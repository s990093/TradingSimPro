# strategies/__init__.py

from mult_strategy.strategies.big.buy_and_hold_strategy import BuyAndHoldStrategy
from mult_strategy.strategies.big.channel_breakout_strategy import ChannelBreakoutStrategy
from .ENV import Environment

from .strategy_manager import StrategyManager

# 2 round


from .big.reference_strategy import MultiReferenceStrategy

from .momentum import strategy_mapping as momentum_mapping
from .overlap import strategy_mapping as overlap_mapping
from .oscillator import strategy_mapping as oscillator_mapping
from .volume import strategy_mapping as volume_mapping
from .trend import strategy_mapping as trend_mapping


__all__ = ['create_strategies']



strategy_mapping ={
            **momentum_mapping,    
            **overlap_mapping,   
            **oscillator_mapping,  
            **volume_mapping,     
            **trend_mapping       
        }
        

# self
strategy_mapping['MultiReferenceStrategy'] = MultiReferenceStrategy
strategy_mapping['BuyAndHoldStrategy'] = BuyAndHoldStrategy
strategy_mapping['ChannelBreakoutStrategy'] = ChannelBreakoutStrategy



def create_strategies():
    strategy_config = Environment.strategy_config
    strategies = []

    # Loop over strategy configurations
    for strategy_info in strategy_config["ta_strategies"]:
        strategy_name = strategy_info["name"]
        strategy_params = strategy_info.get("params")

        # Dynamically create the strategy instance
        strategy_class = strategy_mapping.get(strategy_name)
        if strategy_class:
            strategy_instance = strategy_class(**strategy_params)
            strategies.append(strategy_instance)
        else:
            print(f"Strategy '{strategy_name}' not found.")

    # Create an instance of StrategyManager with the created strategies
    strategy_manager = StrategyManager(strategies)

    return strategy_manager
