from datetime import datetime


class Config:
    # 數據路徑配置
    DATA_RAW_PATH = "data/raw"
    DATA_PROCESSED_PATH = "data/processed"
    
    # 模型配置
    MODEL_SAVE_PATH = "models/saved"
    
    # 日誌配置
    LOG_PATH = "logs"
    
    # 其他配置參數
    DEBUG = True 

class StrategyConfig:
    MAX_THREAD_WORKERS = 30
    MAX_PROCESS_WORKERS = 30
    restart_threshold = 130
    max_restarts = 1000
    # target_stock = "2330.TW"
    # target_stock = "2412.TW"
    target_stock = "2498.TW"
    # 
    start_date='2013-01-01'
    end_date='2022-12-31'
    
    # 策略參數
    BEST_DELTA = 2.15
    
    MAX_POSITION = 2
    INITIAL_UNITS = 5  
    STOP_LOSS_PCT = 0.02  # Δ 值
    initial_capital = 100_000
    
    