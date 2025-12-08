import pytest
from datetime import datetime
import numpy as np
from src.quantum.core.omni_initiative import OmniInitiativeFramework, OmniMetrics

@pytest.fixture
def omni_framework():
    return OmniInitiativeFramework()

def test_omni_framework_initialization(omni_framework):
    """Test initialization of OmniInitiativeFramework"""
    assert omni_framework.metrics_history == []
    assert omni_framework.customer_journeys == {}
    assert omni_framework.employee_workflows == {}
    assert omni_framework.resource_allocations == {}
    assert not omni_framework.is_model_trained
    assert len(omni_framework.performance_targets) == 6

def test_customer_journey_optimization(omni_framework):
    """Test customer journey optimization"""
    journey_data = {
        'satisfaction': 0.8,
        'efficiency': 0.7,
        'revenue': 0.6
    }
    
    optimized_journey = omni_framework.optimize_customer_journey(journey_data)
    
    assert isinstance(optimized_journey, dict)
    assert 'satisfaction' in optimized_journey
    assert 'efficiency' in optimized_journey
    assert 'revenue' in optimized_journey
    assert all(0 <= v <= 1 for v in optimized_journey.values())

def test_employee_workflow_optimization(omni_framework):
    """Test employee workflow optimization"""
    workflow_data = {
        'engagement': 0.7,
        'efficiency': 0.6,
        'satisfaction': 0.8
    }
    
    optimized_workflow = omni_framework.optimize_employee_workflow(workflow_data)
    
    assert isinstance(optimized_workflow, dict)
    assert 'engagement' in optimized_workflow
    assert 'efficiency' in optimized_workflow
    assert 'satisfaction' in optimized_workflow
    assert 'adaptation_rate' in optimized_workflow
    assert all(0 <= v <= 1 for v in optimized_workflow.values() if v != 'adaptation_rate')

def test_resource_optimization(omni_framework):
    """Test resource optimization"""
    current_state = {
        'customer_demand': 0.5,
        'operational_load': 0.5,
        'employee_capacity': 0.5
    }
    
    resource_allocation = omni_framework.optimize_resources(current_state)
    
    assert isinstance(resource_allocation, dict)
    assert 'customer_service' in resource_allocation
    assert 'operations' in resource_allocation
    assert 'employee_support' in resource_allocation
    assert 'data_processing' in resource_allocation
    assert all(0 <= v <= 1 for v in resource_allocation.values())

def test_performance_prediction(omni_framework):
    """Test performance prediction"""
    # Add some metrics history
    for i in range(25):
        metrics = {
            'customer_satisfaction': 0.8 + 0.01 * i,
            'operational_efficiency': 0.7 + 0.01 * i,
            'resource_utilization': 0.6 + 0.01 * i,
            'revenue_impact': 0.5 + 0.01 * i,
            'employee_engagement': 0.7 + 0.01 * i,
            'data_quality': 0.9 + 0.01 * i
        }
        omni_framework.track_metrics(metrics)
    
    # Test prediction
    current_metrics = {
        'customer_satisfaction': 0.85,
        'operational_efficiency': 0.75,
        'resource_utilization': 0.65,
        'revenue_impact': 0.55,
        'employee_engagement': 0.75,
        'data_quality': 0.95
    }
    
    prediction = omni_framework.predict_performance(current_metrics)
    
    assert isinstance(prediction, dict)
    assert 'predicted_satisfaction' in prediction
    assert 'predicted_efficiency' in prediction
    assert 'predicted_utilization' in prediction
    assert 'predicted_revenue' in prediction
    assert 'predicted_engagement' in prediction
    assert 'predicted_data_quality' in prediction
    assert all(0 <= v <= 1 for v in prediction.values())

def test_metrics_tracking(omni_framework):
    """Test metrics tracking"""
    metrics = {
        'customer_satisfaction': 0.85,
        'operational_efficiency': 0.75,
        'resource_utilization': 0.65,
        'revenue_impact': 0.55,
        'employee_engagement': 0.75,
        'data_quality': 0.95
    }
    
    omni_framework.track_metrics(metrics)
    
    assert len(omni_framework.metrics_history) == 1
    assert isinstance(omni_framework.metrics_history[0], OmniMetrics)
    assert omni_framework.metrics_history[0].customer_satisfaction == 0.85
    assert omni_framework.metrics_history[0].operational_efficiency == 0.75
    assert omni_framework.metrics_history[0].resource_utilization == 0.65
    assert omni_framework.metrics_history[0].revenue_impact == 0.55
    assert omni_framework.metrics_history[0].employee_engagement == 0.75
    assert omni_framework.metrics_history[0].data_quality == 0.95

def test_framework_metrics(omni_framework):
    """Test framework metrics retrieval"""
    # Test initial metrics
    initial_metrics = omni_framework.get_framework_metrics()
    assert initial_metrics['customer_satisfaction'] == 0.0
    assert initial_metrics['operational_efficiency'] == 0.0
    assert initial_metrics['resource_utilization'] == 0.0
    assert initial_metrics['revenue_impact'] == 0.0
    assert initial_metrics['employee_engagement'] == 0.0
    assert initial_metrics['data_quality'] == 0.0
    
    # Add some metrics
    metrics = {
        'customer_satisfaction': 0.85,
        'operational_efficiency': 0.75,
        'resource_utilization': 0.65,
        'revenue_impact': 0.55,
        'employee_engagement': 0.75,
        'data_quality': 0.95
    }
    omni_framework.track_metrics(metrics)
    
    # Test updated metrics
    updated_metrics = omni_framework.get_framework_metrics()
    assert updated_metrics['customer_satisfaction'] == 0.85
    assert updated_metrics['operational_efficiency'] == 0.75
    assert updated_metrics['resource_utilization'] == 0.65
    assert updated_metrics['revenue_impact'] == 0.55
    assert updated_metrics['employee_engagement'] == 0.75
    assert updated_metrics['data_quality'] == 0.95
    assert updated_metrics['history_length'] == 1 