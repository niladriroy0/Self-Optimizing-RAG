import pytest

def test_health_endpoint(app_client):
    response = app_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_config_endpoint(app_client):
    response = app_client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "config" in data

def test_query_endpoint(app_client, mocker):
    mock_pipeline = mocker.patch("api.routes.query_routes.rag_pipeline")
    # Return (answer, observability_dict)
    mock_pipeline.return_value = ("Test Answer", {"mode": "test"})
    
    response = app_client.post("/query", json={"question": "What is AI?"})
    
    assert response.status_code == 200
    # The response is a stream, so let's check its content
    content = response.text
    assert "Test Answer" in content
    assert "__OBSERVABILITY_START__" in content
    assert '"mode": "test"' in content

def test_feedback_endpoint(app_client, mocker):
    mock_verify = mocker.patch("api.routes.query_routes.verify_memory")
    mock_verify.return_value = True
    
    response = app_client.post("/feedback", json={"question": "What is AI?", "is_correct": True})
    
    assert response.status_code == 200
    assert response.json()["memory_verified"] is True
    mock_verify.assert_called_once_with("What is AI?", True)
