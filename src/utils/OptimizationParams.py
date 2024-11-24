from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class OptimizationParams:
    severe_loss_threshold: float = 0.0  # 重大虧損閾值
    large_loss_threshold: float = 0.0    # 大額虧損閾值
    medium_loss_threshold: float = 0.0   # 中額虧損閾值
    small_loss_threshold: float = 0.0     # 小額虧損閾值
    small_profit_threshold: float = 0.0   # 小額獲利閾值
    medium_profit_threshold: float = 0.0  # 中額獲利閾值
    large_profit_threshold: float = 0.0   # 大額獲利閾值
    severe_profit_threshold: float = 0.0  # 重大獲利閾值
    
    # 獨立的倉位控制
    large_loss_position: float = 1.0    # 大額虧損倉位
    
    large_profit_position: float = 1.0    # 大額獲利的倉位
    severe_profit_position: float = 1.0   # 重大獲利的倉位

    def display_params(self, path):
        # 參數名稱和對應的值
        params = [
            ("Severe Loss Threshold", self.severe_loss_threshold),
            ("Large Loss Threshold", self.large_loss_threshold),
            ("Medium Loss Threshold", self.medium_loss_threshold),
            ("Small Loss Threshold", self.small_loss_threshold),
            ("Small Profit Threshold", self.small_profit_threshold),
            ("Medium Profit Threshold", self.medium_profit_threshold),
            ("Large Profit Threshold", self.large_profit_threshold),
            ("Severe Profit Threshold", self.severe_profit_threshold),
        ]

        # 倉位名稱和對應的值
        positions = [
            ("Large Loss Position", self.large_loss_position),
            ("Large Profit Position", self.large_profit_position),
            ("Severe Profit Position", self.severe_profit_position),
        ]

        # 分離參數名稱和數值
        names, values = zip(*params)
        pos_names, pos_values = zip(*positions)
        
        # 繪製條形圖
        plt.figure(figsize=(6, 8))  # Adjusted height for two tables
        plt.subplot(2, 1, 1)  # First subplot for Parameters
        plt.barh(names, values, color='skyblue')
        plt.xlabel('Value')
        plt.title('Optimization Parameters')
        plt.grid(axis='x')

        plt.subplot(2, 1, 2)  # Second subplot for Positions
        plt.barh(pos_names, pos_values, color='lightgreen')
        plt.xlabel('Value')
        plt.title('Positions')
        plt.grid(axis='x')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(path)

        # 顯示圖形
        plt.show()
