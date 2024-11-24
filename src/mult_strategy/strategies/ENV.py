import numpy as np
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class Environment(object):
    MAX_THREAD_WORKERS = 30
    MAX_PROCESS_WORKERS = 30
    restart_threshold = 130
    max_restarts = 1000
    target_stock = "2498.TW"
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    CS=60          
    MCN=3
    limit=20
    weights_range = [0, 1, 2, 3, 4, 5]
    x_range = np.arange(0.05, 0.5, 0.05)
    
    adjust_weights = {
        "signal_name" : 'multi_reference_signal',
        "adjustment_factor": 1.5
    }
    
    
    references = [
        {'symbol': '^GSPC', 'ma_window': 100, 'rsi_period': 14},  # S&P 500
        {'symbol': '^DJI', 'ma_window': 100, 'rsi_period': 14},   # Dow Jones
        {'symbol': '^IXIC', 'ma_window': 100, 'rsi_period': 14}   # Nasdaq
    ]
 
    strategy = "ta_strategies"
    strategy_config = {
        "ta_strategies" :[
            # Momentum Strategies
            {"name": "ADXStrategy", "params": {}},
            {"name": "ADXRStrategy", "params": {}},
            {"name": "APOCStrategy", "params": {}},
            {"name": "AROONStrategy", "params": {}},
            {"name": "MACDStrategy", "params": {}},
            {"name": "MACDEXTStrategy", "params": {}},
            {"name": "MFIStrategy", "params": {}},
            {"name": "RSIStrategy", "params": {}},
            {"name": "STOCHStrategy", "params": {}},
            {"name": "WILLRStrategy", "params": {}},
            {"name": "CCIStrategy", "params": {}},
            {"name": "ROCStrategy", "params": {}},
            {"name": "CMOStrategy", "params": {}},
            {"name": "PPOStrategy", "params": {}},

            # Overlap Strategies
            {"name": "BollingerBandsStrategy", "params": {}},
            {"name": "DEMAStrategy", "params": {}},
            {"name": "EMAStrategy", "params": {}},
            {"name": "HilbertTransformTrendlineStrategy", "params": {}},
            {"name": "KAMAStrategy", "params": {}},
            {"name": "MovingAverageStrategy", "params": {}},
            {"name": "MAMAStrategy", "params": {}},
            {"name": "MidPointStrategy", "params": {}},
            {"name": "MidPriceStrategy", "params": {}},
            {"name": "SARStrategy", "params": {}},
            {"name": "SimpleMovingAverageStrategy", "params": {}},
            {"name": "T3Strategy", "params": {}},
            {"name": "WeightedMovingAverageStrategy", "params": {}},
            {"name": "TEMAStrategy", "params": {}},
            {"name": "TRIMAStrategy", "params": {}},

            # Oscillator Strategies
            {"name": "StochasticOscillatorStrategy", "params": {}},
            {"name": "WilliamsRStrategy", "params": {}},
            {"name": "UltimateOscillatorStrategy", "params": {}},

            # Volume Strategies
            {"name": "OBVStrategy", "params": {}},
            {"name": "ChaikinMoneyFlowStrategy", "params": {}},
            {"name": "VolumeRateOfChangeStrategy", "params": {}},
            {"name": "PriceVolumeTrendStrategy", "params": {}},

            # Trend Strategies
            {"name": "ADXTrendStrategy", "params": {}},
            {"name": "IchimokuStrategy", "params": {}},
            {"name": "SuperTrendStrategy", "params": {}},
            {"name": "PABOLStrategy", "params": {}},

            # Other Strategies
            {"name": "ChannelBreakoutStrategy", "params": {}},
            {"name": "BuyAndHoldStrategy", "params": {}},
            {"name": "MultiReferenceStrategy", "params": {"start_date": start_date, "end_date": end_date, "references": references}},

            # # 新增的动量策略
            {"name": "KSTStrategy", "params": {}},
            {"name": "TRIXStrategy", "params": {}},
            {"name": "DPOStrategy", "params": {}},
            {"name": "BullPowerStrategy", "params": {}},
            {"name": "BearPowerStrategy", "params": {}},

            # 新增的重叠策略
            {"name": "VWAPStrategy", "params": {}},
            {"name": "HMAStrategy", "params": {}},
            {"name": "ZLEMAStrategy", "params": {}},
            {"name": "JMAStrategy", "params": {}},

            # 新增的振荡器策略
            {"name": "KDJStrategy", "params": {}},
            {"name": "ElderRayStrategy", "params": {}},
            {"name": "ChaikinOscillatorStrategy", "params": {}},
            {"name": "CoppockCurveStrategy", "params": {}},
            {"name": "RVIStrategy", "params": {}},

            # 新增的成交量策略
            {"name": "EMVStrategy", "params": {}},
            {"name": "FIStrategy", "params": {}},
            {"name": "VWAPMomentumStrategy", "params": {}},
            {"name": "KVOStrategy", "params": {}},
            {"name": "MFVStrategy", "params": {}},

            # 新增的趋势策略
            {"name": "VortexStrategy", "params": {}},
            {"name": "TTMTrendStrategy", "params": {}},
            {"name": "KeltnerChannelStrategy", "params": {}},
            {"name": "MAEnvelopeStrategy", "params": {}},
            {"name": "PSARStrategy", "params": {}},
        ]
    }
    

    
  
    
    @classmethod
    def display_config(cls, signal_columns):
        """Display the configuration of the environment using rich."""
        console = Console()

        # Create a table to display the configuration
        table = Table(title="Environment Configuration")

        # Add columns to the table
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Populate the table with environment properties
        table.add_row("Target Stock", cls.target_stock, style="yellow")
        table.add_row("Start Date", cls.start_date.strftime('%Y-%m-%d'), style="green")
        table.add_row("End Date", cls.end_date.strftime('%Y-%m-%d'),style="yellow")
        table.add_row("CS", str(cls.CS), style="yellow")
        table.add_row("MCN", str(cls.MCN), style="green")
        # table.add_row("Weights Range", str(cls.weights_range))
        # table.add_row("x Range", str(cls.x_range.tolist()))
        # table.add_row("Max Thread Workers", str(cls.MAX_THREAD_WORKERS))
        # table.add_row("Max Process Workers", str(cls.MAX_PROCESS_WORKERS))

        # Display the signal columns
        signal_columns_str = ", ".join(signal_columns)
        table.add_row("Signal Columns", signal_columns_str, style="yellow")

        # Create a panel to contain the table
        panel = Panel(table, title="Configuration Details", border_style="bold blue")

        # Print the panel to the console
        console.print(panel)
