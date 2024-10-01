from fastapi.testclient import TestClient
from src.config.appconfig import ENV
from src.main import app

client = TestClient(app)

base_path = ""

if ENV == "development":
    base_path = "/dev"

def test_health_main():
    response = client.get(f"{base_path}/health")
    assert response.status_code == 200
    assert response.json() == "healthy"

def test_apiHome_main():
    response = client.get(f"{base_path}/api/v1")
    assert response.status_code == 200
    assert response.json() == {
        "ApplicationName": app.title,
        "ApplicationOwner": "GeminiPro AI Agent",
        "ApplicationVersion": "3.0.0",
        "ApplicationEngineer": "Sam Ayo",
        "ApplicationStatus": "running...",
    }
