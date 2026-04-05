import threading
import pytest
from control_plane.config_manager import ConfigManager

def test_config_manager_singleton():
    """Ensure the ConfigManager acts as a true singleton."""
    cm1 = ConfigManager()
    cm2 = ConfigManager()
    assert cm1 is cm2

def test_default_initialization(mock_config_manager):
    """Test that default values are correctly populated."""
    config = mock_config_manager.get_config()
    assert config["top_k"] == 5
    assert config["enable_multi_hop"] is True
    assert mock_config_manager.get_version() == 1

def test_update_config(mock_config_manager):
    """Test full dictionary updates."""
    mock_config_manager.update_config({"top_k": 10, "new_param": "test"})
    assert mock_config_manager.get_param("top_k") == 10
    assert mock_config_manager.get_param("new_param") == "test"
    assert mock_config_manager.get_version() == 2

def test_set_param(mock_config_manager):
    """Test updating a single parameter."""
    mock_config_manager.set_param("chunk_size", 1024)
    assert mock_config_manager.get_param("chunk_size") == 1024
    assert mock_config_manager.get_version() == 2

def test_smart_update(mock_config_manager):
    """Test smart update only increments version if values differ."""
    # Same value, shouldn't increment version
    mock_config_manager.smart_update({"top_k": 5})
    assert mock_config_manager.get_version() == 1

    # New value, should increment version
    mock_config_manager.smart_update({"top_k": 7})
    assert mock_config_manager.get_param("top_k") == 7
    assert mock_config_manager.get_version() == 2

def test_reset_config(mock_config_manager):
    """Test resetting config restores defaults."""
    mock_config_manager.set_param("top_k", 99)
    assert mock_config_manager.get_param("top_k") == 99
    
    mock_config_manager.reset_config()
    assert mock_config_manager.get_param("top_k") == 5

def test_thread_safety(mock_config_manager):
    """Test thread-safe writes do not corrupt state."""
    def worker(val):
        for _ in range(100):
            mock_config_manager.update_config({"test_val": val})

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # The exact value is non-deterministic depending on which thread finished last,
    # but the process should not raise concurrency errors, and version should scale.
    assert mock_config_manager.get_version() >= 100
