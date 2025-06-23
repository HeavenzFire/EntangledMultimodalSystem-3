import pytest
from src import create_app
from src.config import Config

@pytest.fixture
def app():
    app = create_app(Config)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {'status': 'healthy'}

def test_expand_endpoint(client):
    response = client.post('/api/expand', json={'input': [1.0, 2.0, 3.0]})
    assert response.status_code == 200
    assert 'predictions' in response.json

def test_nlp_endpoint(client):
    response = client.post('/api/nlp', json={'prompt': 'Test prompt'})
    assert response.status_code == 200
    assert 'response' in response.json

def test_fractal_endpoint(client):
    response = client.get('/api/fractal')
    assert response.status_code == 200
    assert response.json['status'] == 'success'

def test_radiation_endpoint(client):
    response = client.get('/api/radiation')
    assert response.status_code == 200
    assert 'radiation_data' in response.json 