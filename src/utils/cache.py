import os
import pickle
import hashlib
from typing import Any, Dict


class Cache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.cache_data: Dict[str, Any] = {}  # 使用哈希字串作為 key
        self._create_cache_dir()
        self._load_cache()

    def _create_cache_dir(self):
        """創建快取目錄"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _generate_filename(self, key: str) -> str:
        """生成快取檔案名稱"""
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _load_cache(self):
        """從快取目錄載入資料"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                key = filename[:-4]  # 去掉 `.pkl` 以還原 key
                with open(os.path.join(self.cache_dir, filename), 'rb') as f:
                    self.cache_data[key] = pickle.load(f)

    def get(self, key: str) -> Any:
        """從快取中獲取資料"""
        return self.cache_data.get(key)

    def set(self, key: str, value: Any):
        """將資料存入快取"""
        self.cache_data[key] = value
        self._save_to_cache(key, value)

    def _save_to_cache(self, key: str, value: Any):
        """將資料保存到檔案"""
        filename = self._generate_filename(key)
        with open(filename, 'wb') as f:
            pickle.dump(value, f)
