import pytest
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def test_predict_valid_input_success(mocker, client):
    # Mock the cache
    mock_cache = mocker.Mock()
    mock_cache.get_avg_preparation_time.return_value = 24  # Example average prep time
    client.app.state.cache = mock_cache

    # Mock the model
    mock_model = mocker.Mock()
    mock_model.predict.return_value = 42.0  # Example prediction result
    client.app.state.model = mock_model

    payload = {
        "venue_id": "8a61b55",
        "time_received": "2024-12-02T09:50:01.897036",
        "is_retail": 1,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"delivery_duration": 42.0}


def test_predict_invalid_is_retail(client):
    payload = {
        "venue_id": "8a61b55",
        "time_received": "2024-12-02T09:50:01.897036",
        "is_retail": 2,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Unprocessable Entity
    assert (
        response.json()["detail"][0]["msg"]
        == "Value error, is_retail must be either 0 or 1"
    )


def test_predict_missing_fields(client):
    # Missing 'time_received'
    payload = {
        "venue_id": "8a61b55",
        "is_retail": 1,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Unprocessable Entity
    assert response.json()["detail"][0]["loc"] == ["body", "time_received"]
    assert response.json()["detail"][0]["type"] == "missing"
    assert response.json()["detail"][0]["msg"] == "Field required"


def test_predict_with_extra_fields(client):
    payload = {
        "venue_id": "8a61b55",
        "time_received": "2024-12-02T09:50:01.897036",
        "is_retail": 1,
        "unexpected_field": "unexpected_value",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    assert response.json()["detail"][0]["msg"] == "Extra inputs are not permitted"


def test_predict_invalid_venue_id(mocker, client):
    # Mock the cache
    mock_cache = mocker.Mock()
    mock_cache.get_avg_preparation_time.side_effect = ValueError(
        "Venue ID invalid_id not found in cache."
    )
    client.app.state.cache = mock_cache

    # Mock the model
    mock_model = mocker.Mock()
    mock_model.predict.return_value = 42.0  # Example prediction result
    client.app.state.model = mock_model

    payload = {
        "venue_id": "invalid_id",
        "time_received": "2024-12-02T09:50:01.897036",
        "is_retail": 1,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 400  # Bad request
    assert "Venue ID invalid_id not found in cache." in response.json()["detail"]


def test_predict_invalid_time_received_format(client):
    payload = {
        "venue_id": "8a61b55",
        "time_received": "2024/12/02 09:50:01",  # Incorrect format
        "is_retail": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json()["detail"][0]["loc"] == ["body", "time_received"]
    assert (
        "Input should be a valid datetime or date"
        in response.json()["detail"][0]["msg"]
    )
