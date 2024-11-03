import pandas as pd
import numpy as np


def trades_to_dataframe(trades: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    # Map action codes to descriptive strings
    action_map = {1: 'Buy', 2: 'Sell', 0: 'Stop'}
    # Create DataFrame with columns matching required format
    trades_df = pd.DataFrame({
        'Date': pd.to_datetime(df.index[trades[:, 0].astype(int)]),
        'Action': [action_map[int(action)] for action in trades[:, 1]],
        'Price': trades[:, 2],
    })
    

    return trades_df
