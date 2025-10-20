import pytest
import numpy as np
from src.metaphysical.mathematics.core.quantum_structural import QuantumStructuralSystem
from src.metaphysical.mathematics.core.omni_feedback import OmniFeedbackSystem

class TestBenchmarks:
    @pytest.fixture
    def structural_system(self):
        return QuantumStructuralSystem(num_dimensions=6)
        
    @pytest.fixture
    def feedback_system(self):
        return OmniFeedbackSystem(num_particles=1000)
        
    @pytest.mark.benchmark
    def test_quantum_circuit_benchmark(self, benchmark):
        """Benchmark quantum circuit operations"""
        system = QuantumStructuralSystem(num_dimensions=6)
        metric = np.random.rand(6, 6)
        
        result = benchmark(system.calculate_einstein_tensor, metric)
        assert result.shape == (6, 6)
        
    @pytest.mark.benchmark
    def test_collision_simulation_benchmark(self, benchmark):
        """Benchmark LHC collision simulation"""
        system = OmniFeedbackSystem(num_particles=1000)
        particles = np.random.rand(1000, 1000)
        
        result = benchmark(system.simulate_lhc_collision, particles)
        assert result.shape == (1000, 1000)
        
    @pytest.mark.benchmark
    def test_detector_response_benchmark(self, benchmark):
        """Benchmark detector response simulation"""
        system = OmniFeedbackSystem(num_particles=1000)
        particles = np.random.rand(1000, 1000)
        collision_matrix = system.simulate_lhc_collision(particles)
        
        result = benchmark(system.simulate_detector_response, collision_matrix)
        assert result.shape == (1000, 1000)
        
    @pytest.mark.benchmark
    def test_expansion_optimization_benchmark(self, benchmark):
        """Benchmark expansion optimization"""
        system = OmniFeedbackSystem(num_particles=1000)
        expansion_factor = 0.5
        
        result = benchmark(system.optimize_expansion, expansion_factor)
        assert isinstance(result, float)
        
    @pytest.mark.benchmark
    def test_full_pipeline_benchmark(self, benchmark):
        """Benchmark full processing pipeline"""
        def pipeline():
            structural_system = QuantumStructuralSystem(num_dimensions=6)
            feedback_system = OmniFeedbackSystem(num_particles=1000)
            
            # Process structural integrity
            structural_system.process_structural_integrity()
            
            # Generate particles
            particles = np.concatenate([
                structural_system.state.einstein_tensor.flatten(),
                structural_system.state.calabi_yau_metric.flatten()
            ]).reshape(1000, 1000)
            
            # Process feedback
            feedback_system.process_feedback(particles)
            
            return {
                'structural_status': structural_system.state.system_status,
                'feedback_status': feedback_system.state.system_status
            }
            
        result = benchmark(pipeline)
        assert result['structural_status'] == 'processed'
        assert result['feedback_status'] == 'processed'
        
    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self, benchmark):
        """Benchmark memory usage"""
        import psutil
        import os
        
        def memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
            
        initial_memory = memory_usage()
        
        system = QuantumStructuralSystem(num_dimensions=6)
        system.process_structural_integrity()
        
        final_memory = memory_usage()
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 1000  # Less than 1GB memory increase
        
    @pytest.mark.benchmark
    def test_cpu_usage_benchmark(self, benchmark):
        """Benchmark CPU usage"""
        import psutil
        import os
        
        def cpu_usage():
            process = psutil.Process(os.getpid())
            return process.cpu_percent()
            
        initial_cpu = cpu_usage()
        
        system = OmniFeedbackSystem(num_particles=1000)
        particles = np.random.rand(1000, 1000)
        system.process_feedback(particles)
        
        final_cpu = cpu_usage()
        cpu_increase = final_cpu - initial_cpu
        
        assert cpu_increase < 50  # Less than 50% CPU increase 