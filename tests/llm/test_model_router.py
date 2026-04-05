import pytest
from control_plane.model_router import extract_features, route_by_task, route_model_advanced

def test_extract_features():
    analysis = {
        "text": "def my_func(): pass inside python",
        "length": 10,
        "type": "coding",
        "complexity": "low",
        "ambiguity": 0.3,
        "is_multi_hop": False,
        "needs_keyword": False,
        "needs_semantic": True
    }
    
    features = extract_features(analysis)
    assert features["length"] == 10
    assert features["type"] == "coding"
    assert features["has_code"] == 1
    assert features["requires_coding"] == 1
    assert features["complexity_score"] == 0.3

def test_route_by_task_coding():
    target = route_by_task({"type": "coding", "complexity": "low"})
    assert target == "deepseek-coder:1.3b"

def test_route_by_task_fast_path():
    target = route_by_task({"type": "general", "complexity": "low", "is_multi_hop": False})
    assert target == "phi3:latest"

def test_route_model_advanced_forced(mock_config_manager, mocker):
    mocker.patch("control_plane.model_router.MODEL_PROFILES", {"forced_model": {}})
    mock_config_manager.set_param("model", "forced_model")
    
    res = route_model_advanced({})
    assert res == "forced_model"
