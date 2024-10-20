# strategies/__init__.py

from .buy_and_hold_strategy import BuyAndHoldStrategy
from .channel_breakout_strategy import ChannelBreakoutStrategy
from .strategy_manager import StrategyManager
from .momentum_strategy import MomentumStrategy
from .stochastic_oscillator import StochasticOscillatorStrategy
from .bollinger_band_strategy import BollingerBandsStrategy
from .macd_strategy import MACDStrategy
from .moving_average_strategy import MovingAverageStrategy
from .rsi_strategy import RSIStrategy

# 2 round
from .breakout_strategy import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .stop_loss_strategy import StopLossStrategy
from .trend_following import TrendFollowingStrategy
from .turtle_trading import TurtleTradingStrategy
from .volume_price_strategy import VolumePriceStrategy


__all__ = ['create_strategies']

# 定義各個策略
def create_strategies():
    # long
    moving_average_strategy = MovingAverageStrategy()
    rsi_strategy = RSIStrategy(rsi_period=30)
    macd_strategy = MACDStrategy()
    bollinger_bands_strategy = BollingerBandsStrategy(window=40)
    momentum_strategy = MomentumStrategy()
    
    stochastic_strategy = StochasticOscillatorStrategy()
    
    # 2 round strategies
    breakout_strategy = BreakoutStrategy()
    mean_reversion_strategy = MeanReversionStrategy()
    # import 
    stop_loss_strategy = StopLossStrategy(stop_loss_percent=0.2)
    trend_following_strategy = TrendFollowingStrategy()
    turtle_trading_strategy = TurtleTradingStrategy()
    volume_price_strategy = VolumePriceStrategy()
    
    long_strategies = [
        MovingAverageStrategy(),    
        BollingerBandsStrategy(window=40),
        RSIStrategy(rsi_period=30),
        MACDStrategy(),
        ChannelBreakoutStrategy(),
        BuyAndHoldStrategy()
    ]


    # 將策略添加到策略管理器
    # strategy_manager = StrategyManager([
    #     moving_average_strategy, 
    #     rsi_strategy, 
    #     macd_strategy, 
    #     bollinger_bands_strategy,
    #     momentum_strategy,
    #     stochastic_strategy,
    #     breakout_strategy,
    #     mean_reversion_strategy,
    #     stop_loss_strategy,
    #     trend_following_strategy,
    #     turtle_trading_strategy,
    #     volume_price_strategy,
    # ])
    # 將策略添加到策略管理器
    strategy_manager = StrategyManager(long_strategies)

    return strategy_manager
