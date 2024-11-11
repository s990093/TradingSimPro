from utility.helper.stock_data_cache import StockDataCache
from models.gpt_trader import GPTTrader
from ENV import Environment
import concurrent.futures
import pandas as pd
from typing import List, Tuple
import threading
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

def process_chunk(chunk_data: pd.DataFrame, start_idx: int, window_size: int) -> List[float]:
    gpt_trader = GPTTrader()
    signals = []
    
    for i in range(len(chunk_data)):
        if i + start_idx < window_size:
            signals.append(0)
        else:
            window_data = chunk_data.iloc[max(0, i-window_size):i+1]
            signal = gpt_trader.generate_trading_signal(window_data)
            signals.append(signal)
    
    return signals

def generate_signals_parallel(df_data: pd.DataFrame, window_size: int, num_threads: int = 4) -> List[float]:
    chunk_size = len(df_data) // num_threads
    chunks = []
    
    for i in range(0, len(df_data), chunk_size):
        chunk = df_data.iloc[i:i + chunk_size]
        chunks.append((chunk, i))
    
    all_signals = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("[cyan]生成交易信號...", total=len(chunks))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, start_idx, window_size): (chunk, start_idx)
                for chunk, start_idx in chunks
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_signals = future.result()
                all_signals.extend(chunk_signals)
                progress.advance(task)
    
    return all_signals[:len(df_data)]

def analyze_trades(df_data: pd.DataFrame) -> List[Tuple[str, float, pd.Timestamp]]:
    position = 0
    trades = []
    
    for i in range(len(df_data)):
        signal = df_data['gpt_signal'].iloc[i]
        price = df_data['Close'].iloc[i]
        date = df_data.index[i]
        
        if signal > 0.5 and position <= 0:
            position = 1
            trades.append(('BUY', price, date))
        elif signal < -0.5 and position >= 0:
            position = -1
            trades.append(('SELL', price, date))
    
    return trades

def calculate_returns(trades: List[Tuple[str, float, pd.Timestamp]]) -> float:
    if len(trades) < 2:
        return 0.0
        
    total_return = 1.0
    for i in range(0, len(trades)-1, 2):
        if trades[i][0] == 'BUY':
            buy_price = trades[i][1]
            sell_price = trades[i+1][1]
            trade_return = (sell_price - buy_price) / buy_price
            total_return *= (1 + trade_return)
    
    return total_return - 1

def main():
    # 獲取股票數據
    df_data = StockDataCache(Environment.target_stock, 
                            Environment.start_date, 
                            Environment.end_date).get_data()
    
    window_size = 8
    num_threads = 10  # 可以根據CPU核心數調整
    
    print("開始生成交易信號...")
    signals = generate_signals_parallel(df_data, window_size, num_threads)
    
    # 將信號添加到數據框中
    df_data['gpt_signal'] = signals
    
    # 分析交易
    trades = analyze_trades(df_data)
    
    # 輸出交易結果
    print("\n交易記錄:")
    for trade in trades:
        print(f"{trade[0]} at price {trade[1]:.2f} on {trade[2].strftime('%Y-%m-%d')}")
    
    # 計算並顯示收益率
    total_return = calculate_returns(trades)
    print(f"\n總收益率: {total_return*100:.2f}%")
    
    # 保存結果
    df_data.to_csv('trading_results.csv')
    print("\n結果已保存到 trading_results.csv")

if __name__ == "__main__":
    main() 