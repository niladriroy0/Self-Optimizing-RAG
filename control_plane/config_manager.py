import threading
import copy
from typing import Dict, Any


class ConfigManager:
    """
    Central Control Plane Configuration Manager

    Responsibilities:
    - Maintain current active config
    - Provide read access across system
    - Allow safe updates (thread-safe)
    - Track config versioning (basic)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Singleton pattern to ensure one global config state
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        self._config_lock = threading.Lock()

        # 🔥 Default system config (VERY IMPORTANT)
        self._config: Dict[str, Any] = {
            "top_k": 5,
            "model": None,
            "use_reranker": True,
            "reranker_top_k": 20,
            "prompt_template": "default",
            "chunk_size": 500,
            "temperature": 0.2,
            "max_tokens": 512,
            "enable_multi_hop": True,
            "enable_query_cache": True,
            "confidence_threshold": 0.5,
        }

        # Version tracking (for experiments)
        self._version = 1

    # -------------------------------
    # READ METHODS
    # -------------------------------

    def get_config(self) -> Dict[str, Any]:
        """
        Returns a safe copy of config
        """
        with self._config_lock:
            return copy.deepcopy(self._config)

    def get_param(self, key: str, default=None):
        """
        Get single parameter
        """
        with self._config_lock:
            return self._config.get(key, default)

    def get_version(self) -> int:
        return self._version

    # -------------------------------
    # WRITE METHODS
    # -------------------------------

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update multiple config values safely
        """
        with self._config_lock:
            self._config.update(new_config)
            self._version += 1

    def set_param(self, key: str, value: Any):
        """
        Update single parameter
        """
        with self._config_lock:
            self._config[key] = value
            self._version += 1

    def reset_config(self):
        """
        Reset to default config
        """
        with self._config_lock:
            self._init_config()

    # -------------------------------
    # DEBUG / OBSERVABILITY
    # -------------------------------

    def dump_config(self) -> Dict[str, Any]:
        """
        Returns config with version (useful for logging)
        """
        with self._config_lock:
            return {
                "version": self._version,
                "config": copy.deepcopy(self._config)
            }

    def smart_update(self, new_config: Dict[str, Any]):
        """
        Update only if values actually change
        """
        with self._config_lock:
            updated = False

            for k, v in new_config.items():
                if self._config.get(k) != v:
                    self._config[k] = v
                    updated = True

            if updated:
                self._version += 1

# 🔥 Global access point (IMPORTANT)
config_manager = ConfigManager()