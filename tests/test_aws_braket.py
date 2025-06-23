import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json
import os
from src.integration.aws_braket import AWSBraketIntegration

@pytest.fixture
def aws_braket():
    with patch('boto3.client') as mock_boto3:
        mock_boto3.return_value = Mock()
        integration = AWSBraketIntegration()
        yield integration

def test_initialization(aws_braket):
    """Test AWS Braket integration initialization."""
    assert aws_braket.region == "us-east-1"
    assert aws_braket.qpu_type == "IonQ_Harmony"
    assert aws_braket.qubit_count == 128
    assert aws_braket.error_correction == "surface_code"
    assert aws_braket.state['status'] == 'initialized'

def test_execute_quantum_task(aws_braket):
    """Test quantum task execution."""
    circuit = {
        'qubits': 2,
        'gates': ['H', 'CNOT']
    }
    
    # Mock AWS responses
    aws_braket.braket.create_quantum_task.return_value = {
        'quantumTaskArn': 'test-task-arn'
    }
    
    aws_braket.braket.get_quantum_task.return_value = {
        'status': 'COMPLETED',
        'outputS3Directory': 's3://test-bucket/results'
    }
    
    aws_braket.s3.get_object.return_value = {
        'Body': Mock(read=lambda: json.dumps({'result': 'success'}).encode())
    }
    
    result = aws_braket.execute_quantum_task(circuit)
    
    assert result['task_id'] == 'test-task-arn'
    assert result['result'] == {'result': 'success'}
    assert aws_braket.state['execution_count'] == 1
    assert aws_braket.state['last_execution'] is not None

def test_task_timeout(aws_braket):
    """Test quantum task timeout handling."""
    circuit = {'qubits': 2, 'gates': ['H']}
    
    aws_braket.braket.create_quantum_task.return_value = {
        'quantumTaskArn': 'test-task-arn'
    }
    
    aws_braket.braket.get_quantum_task.return_value = {
        'status': 'RUNNING'
    }
    
    with pytest.raises(TimeoutError):
        aws_braket.execute_quantum_task(circuit, shots=100)

def test_task_failure(aws_braket):
    """Test quantum task failure handling."""
    circuit = {'qubits': 2, 'gates': ['H']}
    
    aws_braket.braket.create_quantum_task.return_value = {
        'quantumTaskArn': 'test-task-arn'
    }
    
    aws_braket.braket.get_quantum_task.return_value = {
        'status': 'FAILED'
    }
    
    with pytest.raises(Exception) as exc_info:
        aws_braket.execute_quantum_task(circuit)
    assert "failed with status: FAILED" in str(exc_info.value)

def test_get_state(aws_braket):
    """Test state retrieval."""
    state = aws_braket.get_state()
    assert state['status'] == 'initialized'
    assert state['execution_count'] == 0
    assert state['error_count'] == 0

def test_get_metrics(aws_braket):
    """Test metrics retrieval."""
    metrics = aws_braket.get_metrics()
    assert metrics['execution_time'] == 0.0
    assert metrics['success_rate'] == 0.0
    assert metrics['error_rate'] == 0.0

def test_reset(aws_braket):
    """Test state and metrics reset."""
    # Modify state and metrics
    aws_braket.state['execution_count'] = 5
    aws_braket.state['error_count'] = 2
    aws_braket.metrics['execution_time'] = 10.5
    
    aws_braket.reset()
    
    assert aws_braket.state['execution_count'] == 0
    assert aws_braket.state['error_count'] == 0
    assert aws_braket.metrics['execution_time'] == 0.0 