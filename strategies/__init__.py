# strategies/__init__.py

from ENV import Environment
from strategies.strategy_manager import StrategyManager
from .buy_and_hold_strategy import BuyAndHoldStrategy
from .channel_breakout_strategy import ChannelBreakoutStrategy
# from .strategy_manager import StrategyManager
# from .momentum_strategy import MomentumStrategy
# from .stochastic_oscillator import StochasticOscillatorStrategy
# from .bollinger_band_strategy import BollingerBandsStrategy
# from .macd_strategy import MACDStrategy
# from .moving_average_strategy import MovingAverageStrategy
# from .rsi_strategy import RSIStrategy

# 2 round
# from .breakout_strategy import BreakoutStrategy
# from .mean_reversion import MeanReversionStrategy
# from .stop_loss_strategy import StopLossStrategy
# from .trend_following import TrendFollowingStrategy
# from .turtle_trading import TurtleTradingStrategy
# from .volume_price_strategy import VolumePriceStrategy


from .big.reference_strategy import MultiReferenceStrategy

from .momentum  import strategy_mapping as momentum_strategy_mapping
from .overlap  import strategy_mapping as overlap_strategy_mapping

__all__ = ['create_strategies']




strategy_mapping = {**momentum_strategy_mapping, **overlap_strategy_mapping}
strategy_mapping['MultiReferenceStrategy'] = MultiReferenceStrategy
strategy_mapping['BuyAndHoldStrategy'] = BuyAndHoldStrategy
strategy_mapping['ChannelBreakoutStrategy'] = ChannelBreakoutStrategy



# strategy_mapping = {
#     "MovingAverageStrategy": MovingAverageStrategy,
#     "BollingerBandsStrategy": BollingerBandsStrategy,
#     "RSIStrategy": RSIStrategy,
#     "MACDStrategy": MACDStrategy,
#     "ChannelBreakoutStrategy": ChannelBreakoutStrategy,
#     "BuyAndHoldStrategy": BuyAndHoldStrategy,
#     "MultiReferenceStrategy": MultiReferenceStrategy,
#     "MomentumStrategy": MomentumStrategy,
#     "StochasticOscillatorStrategy": StochasticOscillatorStrategy,
#     "BreakoutStrategy": BreakoutStrategy,
#     "MeanReversionStrategy": MeanReversionStrategy,
#     "StopLossStrategy": StopLossStrategy,
#     "TrendFollowingStrategy": TrendFollowingStrategy,
#     "TurtleTradingStrategy": TurtleTradingStrategy,
#     "VolumePriceStrategy": VolumePriceStrategy,
# }

def  create_strategies():
    strategy_config = Environment.strategy_config
    strategies = []

    # Loop over strategy configurations
    for strategy_info in strategy_config[Environment.strategy]:
        strategy_name = strategy_info["name"]
        strategy_params = strategy_info.get("params", {})

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
