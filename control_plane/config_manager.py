import threading
import copy
import yaml
from pathlib import Path
from typing import Dict, Any

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "system_config.yaml"

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

        # 🔥 DEFAULT SYSTEM CONFIG
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
        
        # Override defaults with any values saved in YAML
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads configuration from YAML file if it exists."""
        if _CONFIG_PATH.exists():
            try:
                with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                    disk_config = yaml.safe_load(f)
                    if disk_config and isinstance(disk_config, dict):
                        self._config.update(disk_config)
            except Exception as e:
                pass # Fallback to defaults on error
    
    def _save_to_disk(self):
        """Saves current configuration to YAML file. Must be called under lock."""
        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            pass

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
            self._save_to_disk()

    def set_param(self, key: str, value: Any):
        with self._config_lock:
            self._config[key] = value
            self._version += 1
            self._save_to_disk()

    def reset_config(self):
        with self._config_lock:
            self._init_config()
            self._save_to_disk()

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
                self._save_to_disk()

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