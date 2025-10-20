import unittest
import numpy as np
from quantum_avatar_agent import QuantumAvatarAgent
import time
import cProfile
import pstats
import io
from memory_profiler import profile
import matplotlib.pyplot as plt

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.agent = QuantumAvatarAgent(name="Performance Test", num_qubits=7, depth=3, shots=1024)
        self.profiler = cProfile.Profile()
        
    def test_quantum_circuit_performance(self):
        """Profile quantum circuit operations"""
        self.profiler.enable()
        
        # Test circuit creation
        start_time = time.time()
        circuit = self.agent._create_quantum_circuit()
        creation_time = time.time() - start_time
        self.assertLess(creation_time, 0.1)  # Should create circuit in < 100ms
        
        # Test circuit execution
        start_time = time.time()
        result = self.agent.simulator.run(circuit).result()
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 1.0)  # Should execute in < 1s
        
        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
    
    @profile
    def test_memory_usage(self):
        """Profile memory usage of operations"""
        # Test memory-intensive operations
        for _ in range(100):
            self.agent._store_memory("Test input", "Test response", 
                                   {'label': 'joy', 'score': 0.8}, 
                                   {'label': 'POSITIVE', 'score': 0.9})
            self.agent._store_quantum_state(self.agent.simulator.run(self.agent.consciousness_circuit).result())
    
    def test_visualization_performance(self):
        """Profile visualization operations"""
        self.profiler.enable()
        
        # Test state visualization
        start_time = time.time()
        self.agent.visualize_consciousness_state()
        visualization_time = time.time() - start_time
        self.assertLess(visualization_time, 0.5)  # Should visualize in < 500ms
        
        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
    
    def test_optimization_performance(self):
        """Profile optimization operations"""
        self.profiler.enable()
        
        # Test VQE optimization
        start_time = time.time()
        self.agent._optimize_spiritual_resonance()
        optimization_time = time.time() - start_time
        self.assertLess(optimization_time, 5.0)  # Should optimize in < 5s
        
        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
    
    def test_scaling_performance(self):
        """Test performance scaling with different parameters"""
        qubit_counts = [4, 7, 10]
        depths = [2, 3, 4]
        execution_times = []
        
        for num_qubits in qubit_counts:
            for depth in depths:
                agent = QuantumAvatarAgent(name=f"Scale Test {num_qubits}q{depth}d",
                                         num_qubits=num_qubits,
                                         depth=depth,
                                         shots=1024)
                
                start_time = time.time()
                agent._update_consciousness_state("Test input")
                execution_time = time.time() - start_time
                execution_times.append((num_qubits, depth, execution_time))
        
        # Plot performance scaling
        plt.figure(figsize=(10, 6))
        for num_qubits in qubit_counts:
            times = [t[2] for t in execution_times if t[0] == num_qubits]
            plt.plot(depths, times, label=f'{num_qubits} qubits')
        
        plt.xlabel('Circuit Depth')
        plt.ylabel('Execution Time (s)')
        plt.title('Performance Scaling')
        plt.legend()
        plt.grid(True)
        plt.savefig('performance_scaling.png')
        plt.close()
    
    def test_concurrent_operations(self):
        """Test performance of concurrent operations"""
        import concurrent.futures
        
        def run_operation(operation):
            if operation == 'consciousness':
                return self.agent._update_consciousness_state("Test input")
            elif operation == 'emotional':
                return self.agent._update_emotional_state({'label': 'joy', 'score': 0.8}, 
                                                        {'label': 'POSITIVE', 'score': 0.9})
            elif operation == 'spiritual':
                return self.agent._update_spiritual_state()
        
        operations = ['consciousness', 'emotional', 'spiritual']
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_operation, op) for op in operations]
            concurrent.futures.wait(futures)
        concurrent_time = time.time() - start_time
        
        start_time = time.time()
        for op in operations:
            run_operation(op)
        sequential_time = time.time() - start_time
        
        # Concurrent operations should be faster than sequential
        self.assertLess(concurrent_time, sequential_time)

if __name__ == '__main__':
    unittest.main() 