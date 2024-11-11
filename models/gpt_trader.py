import pandas as pd
from utils.gpt_analyzer import analyze_with_gpt

class GPTTrader:
    def __init__(self):
        self.system_prompt = """
            你是一個專業的量化交易分析師。請基於提供的技術數據進行分析並給出明確的交易信號。
            規則：
            1. 必須只返回一個介於 -1 到 1 之間的數字
            2. 信號強度說明：
            * 1.0 = 強烈買入
            * 0.5 = 建議買入
            * 0.0 = 持平/觀望
            * -0.5 = 建議賣出
            * -1.0 = 強烈賣出
            3. 請根據價格趨勢、成交量變化等技術指標綜合判斷
            4. 嚴格返回數字，不要包含任何其他文字
        """
        
    def generate_trading_signal(self, stock_data: pd.DataFrame, window_size: int = 5) -> float:
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
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        volume_change = (data['Volume'].iloc[-1] - data['Volume'].iloc[0]) / data['Volume'].iloc[0] * 100
        avg_volume = data['Volume'].mean()
        price_trend = "上升" if price_change > 0 else "下降"
        
        return f"""技術分析數據：

            1. 價格指標：
            - 最新收盤價：{data['Close'].iloc[-1]:.2f}
            - {len(data)}日價格變化：{price_change:+.2f}%
            - 價格趨勢：{price_trend}
            - 最高價：{data['High'].max():.2f}
            - 最低價：{data['Low'].min():.2f}

            2. 成交量指標：
            - 成交量變化：{volume_change:+.2f}%
            - 平均成交量：{avg_volume:.0f}

            3. 歷史數據（近{len(data)}日）：
            {data[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()}

            請根據以上數據給出介於-1到1之間的交易信號數字："""
    
    def _parse_response(self, response: str) -> float:
        
        try:
            cleaned_response = response.strip()
     
                
            signal = float(cleaned_response)
          
            return max(min(signal, 1.0), -1.0)
            
        except ValueError as e:
            print(f"Error parsing response: {e}")
            # 請求 GPT 重新檢查並修正回應
            verification_prompt = """
            請檢查以下回應是否符合要求：必須是一個介於 -1 到 1 之間的數字。
            如果不符合要求，請直接返回一個符合要求的數字。
            
            原始回應：
            {response}
            
            請只返回一個介於 -1 到 1 之間的數字："""
            
            retry_response = analyze_with_gpt(
                content=verification_prompt.format(response=response),
                task_prompt=self.system_prompt,
                temperature=0.2,
                max_tokens=10
            )
            
            # 遞迴調用解析函數處理重試結果
            return self._parse_response(retry_response)