"""
Tests for Flask application routes.
"""
import pytest
from app import app


@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'

    def test_health_json_format(self, client):
        response = client.get('/health')
        data = response.get_json()
        assert 'status' in data
        assert 'model_loaded' in data


class TestIndexPage:
    def test_index_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_index_contains_title(self, client):
        response = client.get('/')
        assert b'STCA-Net' in response.data


class TestDetectPage:
    def test_detect_get_returns_200(self, client):
        response = client.get('/detect')
        assert response.status_code == 200

    def test_detect_post_no_file(self, client):
        response = client.post('/detect')
        assert response.status_code == 200
        assert b'No video file uploaded' in response.data


class TestImageDetectPage:
    def test_image_detect_get_returns_200(self, client):
        response = client.get('/image-detect')
        assert response.status_code == 200

    def test_image_detect_post_no_file(self, client):
        response = client.post('/image-detect')
        assert response.status_code == 200
        assert b'No image file uploaded' in response.data
