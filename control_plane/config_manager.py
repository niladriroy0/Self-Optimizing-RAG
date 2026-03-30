import threading
import copy
from typing import Dict, Any


class ConfigManager:

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        self._config_lock = threading.Lock()

        # 🔥 FINAL SYSTEM CONFIG
        self._config: Dict[str, Any] = {

            # -----------------------
            # RETRIEVAL
            # -----------------------
            "top_k": 5,
            "use_reranker": True,
            "reranker_top_k": 20,
            "chunk_size": 500,

            # -----------------------
            # MODEL CONTROL
            # -----------------------
            "model": None,
            "temperature": 0.2,
            "max_tokens": 512,

            # -----------------------
            # INTELLIGENCE FEATURES
            # -----------------------
            "enable_multi_hop": True,
            "enable_decomposition": True,   # 🔥 NEW
            "enable_fallback": True,        # 🔥 NEW

            # -----------------------
            # HYBRID RETRIEVAL
            # -----------------------
            "enable_hybrid": True,
            "keyword_weight": 0.5,
            "semantic_weight": 0.5,

            # -----------------------
            # MEMORY / CACHE
            # -----------------------
            "enable_query_cache": True,
            "enable_memory": True,
            "confidence_threshold": 0.6,

            # -----------------------
            # OBSERVABILITY
            # -----------------------
            "enable_logging": True,
        }

        self._version = 1

    # -------------------------------
    # READ METHODS
    # -------------------------------

    def get_config(self) -> Dict[str, Any]:
        with self._config_lock:
            return copy.deepcopy(self._config)

    def get_param(self, key: str, default=None):
        with self._config_lock:
            return self._config.get(key, default)

    def get_version(self) -> int:
        return self._version

    # -------------------------------
    # WRITE METHODS
    # -------------------------------

    def update_config(self, new_config: Dict[str, Any]):
        with self._config_lock:
            self._config.update(new_config)
            self._version += 1

    def set_param(self, key: str, value: Any):
        with self._config_lock:
            self._config[key] = value
            self._version += 1

    def reset_config(self):
        with self._config_lock:
            self._init_config()

    # -------------------------------
    # SMART UPDATE (NO NOISE)
    # -------------------------------

    def smart_update(self, new_config: Dict[str, Any]):

        with self._config_lock:
            updated = False

            for k, v in new_config.items():
                if self._config.get(k) != v:
                    self._config[k] = v
                    updated = True

            if updated:
                self._version += 1

    # -------------------------------
    # DEBUG / OBSERVABILITY
    # -------------------------------

    def dump_config(self) -> Dict[str, Any]:

        with self._config_lock:
            return {
                "version": self._version,
                "config": copy.deepcopy(self._config)
            }


# 🔥 GLOBAL INSTANCE
config_manager = ConfigManager()