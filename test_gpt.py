from utility.helper.stock_data_cache import StockDataCache
import openai
import pandas as pd
import random
from typing import List, Dict
from datetime import datetime, timedelta
from ENV import Environment


api_keys = ["sk-JsX1k5W4cJmkII0F005273A2E4D1430eBf1bB38a92Dc33Fb",
       "sk-ouapzZyTJaNSUAO480744a8693F446B1B1A552Da73A05971",
       "sk-IjlK3saoqmSqp22f3842B88e8fCd44F0AfD2D8E841C9FaF7"
       ]

openai.base_url = "https://free.v36.cm"
openai.default_headers = {"x-foo": "true"}


def analyze_with_gpt(content, task_prompt, model="gpt-3.5-turbo-0125", temperature=0, 
                    max_tokens=200, top_p=1, frequency_penalty=0.0, presence_penalty=0.0):
    """
    使用 OpenAI GPT 模型根據給定內容和任務提示生成回應。
    """
    openai.api_key ="sk-ouapzZyTJaNSUAO480744a8693F446B1B1A552Da73A05971"

    try:
        completion = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=[
                {"role": "system", "content": task_prompt},
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Error in generating response"

class GPTTrader:
    def __init__(self):
        self.system_prompt = """你是一個專業的交易分析師。請根據提供的股票數據提供交易建議。
        請只回答一個數字，範圍從 -1 到 1：
        - 1 表示強烈買入信號
        - -1 表示強烈賣出信號
        - 0 表示持平
        中間值表示相應強度"""
        
    def generate_trading_signal(self, stock_data: pd.DataFrame, window_size: int = 5) -> float:
        """使用 GPT 來分析最近的股票數據並生成交易信號"""
        recent_data = stock_data.tail(window_size)
        data_description = self._prepare_data_description(recent_data)
        
        response = analyze_with_gpt(
            content=data_description,
            task_prompt=self.system_prompt,
            temperature=0.3,
            max_tokens=50
        )
        
        return self._parse_response(response)
    
    def _prepare_data_description(self, data: pd.DataFrame) -> str:
        """準備股票數據描述"""
        # 計算一些基本指標
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        volume_change = (data['Volume'].iloc[-1] - data['Volume'].iloc[0]) / data['Volume'].iloc[0] * 100
        
        return f"""
        請分析以下股票數據並給出交易信號：
        
        最新收盤價: {data['Close'].iloc[-1]:.2f}
        價格變化百分比: {price_change:.2f}%
        成交量變化百分比: {volume_change:.2f}%
        
        最近{len(data)}天數據：
        {data[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()}
        """
    
    def _parse_response(self, response: str) -> float:
        """解析 GPT 的回應並轉換為信號值"""
        try:
            # 提取數值
            signal = float(response.strip())
            # 確保信號在 -1 到 1 之間
            return max(min(signal, 1.0), -1.0)
        except:
            return 0.0

def main():
    # 獲取股票數據
    df_data = StockDataCache(Environment.target_stock, 
                            Environment.start_date, 
                            Environment.end_date).get_data()
    
    # 初始化 GPT Trader
    gpt_trader = GPTTrader()
    
    # 生成交易信號
    signals = []
    window_size = 5
    
    print("開始生成交易信號...")
    for i in range(len(df_data)):
        if i < window_size:
            signals.append(0)
        else:
            window_data = df_data.iloc[i-window_size:i]
            signal = gpt_trader.generate_trading_signal(window_data)
            signals.append(signal)
            if i % 10 == 0:  # 每10次迭代顯示進度
                print(f"處理進度: {i}/{len(df_data)}")
    
    # 將信號添加到數據框中
    df_data['gpt_signal'] = signals
    
    # 模擬交易
    position = 0
    trades = []
    
    for i in range(len(df_data)):
        signal = df_data['gpt_signal'].iloc[i]
        price = df_data['Close'].iloc[i]
        date = df_data.index[i]
        
        if signal > 0.5 and position <= 0:  # 買入信號
            position = 1
            trades.append(('BUY', price, date))
        elif signal < -0.5 and position >= 0:  # 賣出信號
            position = -1
            trades.append(('SELL', price, date))
    
    # 輸出交易結果
    print("\n交易記錄:")
    for trade in trades:
        print(f"{trade[0]} at price {trade[1]:.2f} on {trade[2].strftime('%Y-%m-%d')}")
    
    # 計算收益率
    if len(trades) >= 2:
        total_return = 1.0
        for i in range(0, len(trades)-1, 2):
            if trades[i][0] == 'BUY':
                buy_price = trades[i][1]
                sell_price = trades[i+1][1]
                trade_return = (sell_price - buy_price) / buy_price
                total_return *= (1 + trade_return)
        
        print(f"\n總收益率: {(total_return-1)*100:.2f}%")
    
    # 保存結果
    df_data.to_csv('trading_results.csv')
    print("\n結果已保存到 trading_results.csv")

if __name__ == "__main__":
    main()
