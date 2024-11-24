import itertools
import numpy as np

def adjust_weights(weights, signal_name, adjustment_factor, signal_columns):
    
    # 查找信号名称的索引
    if signal_name in signal_columns:
        index = signal_columns.index(signal_name)
    else:
        raise ValueError(f"Signal '{signal_name}' not found in signal names.")
    
    # 检查 weights 是否包含 None，并替换为默认值（如 0 或 1），或抛出错误
    if weights[index] is None:
        raise ValueError(f"Weight for signal '{signal_name}' is None.")

    # 计算新的权重
    new_weights = weights.copy()
    
    # 调整权重
    new_weights[index] *= adjustment_factor
    
    # 确保权重总和为 1
    total_weight = sum(new_weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero after adjustment.")
    
    new_weights = [weight / total_weight for weight in new_weights]
    
    return new_weights

