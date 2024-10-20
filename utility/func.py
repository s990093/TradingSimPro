# def save_data():
#         # Prepare data to save
#     results = {
#         "target_stock": Environment.target_stock,  # Ensure this variable is defined in your code
#         "best_weights": list(best_bee['weights']),  # Save as a list
#         "best_threshold": float(best_bee['x']),  # Ensure it's a float for JSON serialization
#         "best_fitness": best_fitness,  # Ensure this variable is defined
#         # Convert DataFrame to a dictionary for JSON serialization
#         "df_data": df_data.to_dict(orient='records'),  # Converts DataFrame to a list of records
#     }
    
#     # Save results to JSON file
#     with open('strategy_results.json', 'w') as json_file:
#         json.dump(results, json_file, indent=4)


def adjust_weights(weights, signal_name, adjustment_factor, signal_names):
    # 查找信号名称的索引
    if signal_name in signal_names:
        index = signal_names.index(signal_name)
    else:
        raise ValueError(f"Signal '{signal_name}' not found in signal names.")

    # 计算新的权重
    new_weights = weights.copy()
    new_weights[index] *= adjustment_factor
    
    # 确保权重总和为 1
    total_weight = sum(new_weights)
    new_weights = [weight / total_weight for weight in new_weights]
    
    return new_weights

