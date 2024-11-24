class StrategyParameters:
    TREND_WINDOWS = {
        'short': 5,
        'medium': 10,
        'long': 20
    }
    
    MOMENTUM_PARAMS = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    VOLUME_PARAMS = {
        'ma_period': 20,
        'std_period': 20
    }
    
    ENV_PARAMS = {
        'window': 10,
        'up_factor': 1.0,
        'down_factor': 1.0
    } 