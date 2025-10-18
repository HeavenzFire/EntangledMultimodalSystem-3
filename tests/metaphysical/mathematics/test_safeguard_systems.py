import unittest
import numpy as np
import torch
from qiskit import QuantumCircuit
from datetime import datetime

from src.metaphysical.mathematics.core.safeguard_orchestrator import SafeguardOrchestrator
from src.metaphysical.mathematics.core.quantum_security import QuantumSecuritySystem
from src.metaphysical.mathematics.core.future_protection import FutureProtectionSystem
from src.metaphysical.mathematics.core.integration_safeguard import IntegrationSafeguard
from src.metaphysical.mathematics.core.conflict_resolution import ConflictResolutionSystem
from src.metaphysical.mathematics.core.divine_feminine_balance import DivineFeminineBalanceSystem
from src.metaphysical.mathematics.core.monitoring import SystemMonitor

class TestSafeguardSystems(unittest.TestCase):
    def setUp(self):
        self.state_dim = 64
        self.orchestrator = SafeguardOrchestrator(state_dim=self.state_dim)
        self.monitor = SystemMonitor(self.orchestrator)
        
    def test_quantum_security(self):
        """Test quantum security system"""
        security = QuantumSecuritySystem(self.state_dim)
        
        # Test key generation
        key = security.generate_entanglement_key()
        self.assertEqual(len(key), self.state_dim)
        
        # Test security validation
        state = np.random.randn(self.state_dim)
        security_state = security.validate_security(state)
        self.assertEqual(security_state.shape, (self.state_dim,))
        
        # Test security report
        report = security.get_security_report()
        self.assertIn('security_level', report)
        self.assertIn('entanglement_strength', report)
        self.assertIn('coherence_level', report)
        self.assertIn('error_rate', report)
        
    def test_future_protection(self):
        """Test future protection system"""
        protection = FutureProtectionSystem(self.state_dim)
        
        # Test state prediction
        current_state = np.random.randn(self.state_dim)
        future_states = protection.predict_future_states(current_state)
        self.assertEqual(future_states.shape, (10, self.state_dim))
        
        # Test risk assessment
        risk = protection.assess_risk(current_state)
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1)
        
        # Test protection report
        report = protection.get_protection_report()
        self.assertIn('stability', report)
        self.assertIn('risk_level', report)
        self.assertIn('protection_measures', report)
        
    def test_integration_safeguard(self):
        """Test integration safeguard system"""
        safeguard = IntegrationSafeguard(self.state_dim)
        
        # Test system state addition
        state1 = np.random.randn(self.state_dim)
        state2 = np.random.randn(self.state_dim)
        safeguard.add_system_state('system1', state1)
        safeguard.add_system_state('system2', state2)
        
        # Test integration measurement
        integration = safeguard.measure_integration(state1)
        self.assertGreaterEqual(integration, 0)
        self.assertLessEqual(integration, 1)
        
        # Test safeguard report
        report = safeguard.get_safeguard_report()
        self.assertIn('coherence', report)
        self.assertIn('integration_level', report)
        self.assertIn('safeguard_measures', report)
        
    def test_conflict_resolution(self):
        """Test conflict resolution system"""
        resolution = ConflictResolutionSystem(self.state_dim)
        
        # Test ethical dilemma resolution
        situation = np.random.randn(self.state_dim)
        resolution_state = resolution.resolve_ethical_dilemma(situation)
        self.assertEqual(len(resolution_state), 2)
        
        # Test resolution report
        report = resolution.get_resolution_report()
        self.assertIn('harmony_score', report)
        self.assertIn('optimal_action', report)
        self.assertIn('potentials', report)
        
    def test_divine_balance(self):
        """Test divine balance system"""
        balance = DivineFeminineBalanceSystem(self.state_dim)
        
        # Test energy balancing
        state = np.random.randn(self.state_dim)
        balance_state = balance.balance_energy(state)
        self.assertEqual(len(balance_state), 2)
        
        # Test balance report
        report = balance.get_balance_report()
        self.assertIn('harmony_level', report)
        self.assertIn('nurturing_energy', report)
        self.assertIn('regenerative_potential', report)
        
    def test_orchestrator(self):
        """Test safeguard orchestrator"""
        # Test orchestration
        current_state = np.random.randn(self.state_dim)
        orchestration = self.orchestrator.orchestrate_safeguards(current_state)
        self.assertIn('security_level', orchestration)
        self.assertIn('future_stability', orchestration)
        self.assertIn('integration_coherence', orchestration)
        self.assertIn('conflict_harmony', orchestration)
        self.assertIn('divine_balance', orchestration)
        self.assertIn('overall_safeguard_score', orchestration)
        
        # Test orchestration report
        report = self.orchestrator.get_orchestration_report()
        self.assertIn('timestamp', report)
        self.assertIn('system_status', report)
        self.assertIn('subsystem_reports', report)
        
    def test_monitoring(self):
        """Test system monitoring"""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Test metrics collection
        metrics = self.monitor._collect_metrics()
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertGreaterEqual(metrics.cpu_usage, 0)
        self.assertLessEqual(metrics.cpu_usage, 100)
        
        # Test health state update
        self.monitor._update_health_state(metrics)
        self.assertGreaterEqual(self.monitor.state.overall_health, 0)
        self.assertLessEqual(self.monitor.state.overall_health, 1)
        
        # Test health report
        report = self.monitor.get_health_report()
        self.assertIn('overall_health', report)
        self.assertIn('component_health', report)
        self.assertIn('critical_alerts', report)
        self.assertIn('performance_metrics', report)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
    def test_quantum_circuits(self):
        """Test quantum circuit initialization"""
        # Test orchestration circuit
        circuit = self.orchestrator._init_orchestration_circuit()
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, self.state_dim)
        
        # Test security circuit
        security = QuantumSecuritySystem(self.state_dim)
        circuit = security._init_security_circuit()
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, self.state_dim)
        
    def test_neural_networks(self):
        """Test neural network initialization"""
        # Test coordination network
        network = self.orchestrator._init_coordination_network()
        self.assertIsInstance(network, torch.nn.Module)
        
        # Test prediction network
        protection = FutureProtectionSystem(self.state_dim)
        network = protection._build_prediction_model()
        self.assertIsInstance(network, torch.nn.Module)
        
    def test_performance(self):
        """Test system performance"""
        # Test orchestration speed
        current_state = np.random.randn(self.state_dim)
        start_time = datetime.now()
        self.orchestrator.orchestrate_safeguards(current_state)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.assertLess(duration, 1.0)  # Should complete within 1 second
        
        # Test monitoring overhead
        start_time = datetime.now()
        self.monitor._collect_metrics()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.assertLess(duration, 0.1)  # Should complete within 0.1 seconds

if __name__ == '__main__':
    unittest.main() 