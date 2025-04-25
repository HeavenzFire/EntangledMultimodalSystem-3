#!/usr/bin/env python3
"""
Quantum Time Dilation Demo
=========================
This script demonstrates the capabilities of the Quantum Time Dilation framework
for accelerating quantum computations.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from quantum_time_dilation import QuantumTimeDilation

def create_complex_circuit(num_qubits=4, depth=10):
    """Create a complex quantum circuit with multiple gates."""
    qc = QuantumCircuit(num_qubits)
    
    # Apply Hadamard gates to create superposition
    qc.h(range(num_qubits))
    
    # Apply a series of gates to create entanglement
    for i in range(depth):
        # Apply random single-qubit gates
        for qubit in range(num_qubits):
            gate_type = np.random.choice(['h', 'x', 'y', 'z', 's', 't'])
            if gate_type == 'h':
                qc.h(qubit)
            elif gate_type == 'x':
                qc.x(qubit)
            elif gate_type == 'y':
                qc.y(qubit)
            elif gate_type == 'z':
                qc.z(qubit)
            elif gate_type == 's':
                qc.s(qubit)
            elif gate_type == 't':
                qc.t(qubit)
        
        # Apply two-qubit gates to create entanglement
        for j in range(num_qubits-1):
            if np.random.random() > 0.5:
                qc.cx(j, j+1)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc

def compare_execution_times():
    """Compare execution times with and without time dilation."""
    # Parameters
    num_qubits = 4
    circuit_depth = 15
    target_time = 1.0
    
    print("Creating complex quantum circuit...")
    complex_circuit = create_complex_circuit(num_qubits, circuit_depth)
    print(complex_circuit)
    
    # Standard execution (without time dilation)
    print("\nExecuting circuit without time dilation...")
    start_time = time.time()
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(complex_circuit, simulator).result()
    standard_time = time.time() - start_time
    print(f"Standard execution time: {standard_time:.4f} seconds")
    
    # Time dilation execution
    print("\nExecuting circuit with Quantum Time Dilation...")
    qtd = QuantumTimeDilation(
        num_qubits=num_qubits,
        num_streams=20,  # Reduced for demonstration
        base_acceleration=10.0,  # Reduced for demonstration
        predictive_depth=5,
        adaptive_rate=0.1,
        coherence_threshold=0.95
    )
    
    start_time = time.time()
    results = qtd.accelerate_computation(complex_circuit, target_time)
    dilated_time = time.time() - start_time
    
    # Calculate speedup
    speedup = standard_time / dilated_time
    
    print(f"Time dilation execution time: {dilated_time:.4f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")
    print(f"Virtual time reached: {results['virtual_time_reached']:.4f}")
    print(f"Average performance: {results['average_performance']:.4f}")
    print(f"Average coherence: {results['average_coherence']:.4f}")
    
    # Visualize results
    qtd.visualize_results(results)
    qtd.visualize_metrics()
    
    return speedup, results

def demonstrate_time_dilation_scaling():
    """Demonstrate how time dilation scales with different parameters."""
    # Parameters to test
    stream_counts = [5, 10, 20, 50]
    acceleration_factors = [1.0, 5.0, 10.0, 20.0]
    
    # Results storage
    speedups = np.zeros((len(stream_counts), len(acceleration_factors)))
    
    # Create a complex circuit
    complex_circuit = create_complex_circuit(4, 10)
    
    # Standard execution time (baseline)
    simulator = Aer.get_backend('statevector_simulator')
    start_time = time.time()
    result = execute(complex_circuit, simulator).result()
    standard_time = time.time() - start_time
    
    print(f"Standard execution time: {standard_time:.4f} seconds")
    
    # Test different combinations
    for i, num_streams in enumerate(stream_counts):
        for j, acc_factor in enumerate(acceleration_factors):
            print(f"\nTesting with {num_streams} streams and acceleration factor {acc_factor}...")
            
            qtd = QuantumTimeDilation(
                num_qubits=4,
                num_streams=num_streams,
                base_acceleration=acc_factor,
                predictive_depth=5,
                adaptive_rate=0.1,
                coherence_threshold=0.95
            )
            
            start_time = time.time()
            results = qtd.accelerate_computation(complex_circuit, target_time=1.0)
            dilated_time = time.time() - start_time
            
            speedup = standard_time / dilated_time
            speedups[i, j] = speedup
            
            print(f"Execution time: {dilated_time:.4f} seconds")
            print(f"Speedup factor: {speedup:.2f}x")
    
    # Visualize scaling results
    plt.figure(figsize=(10, 8))
    for i, num_streams in enumerate(stream_counts):
        plt.plot(acceleration_factors, speedups[i], marker='o', label=f"{num_streams} streams")
    
    plt.xlabel("Acceleration Factor")
    plt.ylabel("Speedup Factor")
    plt.title("Time Dilation Scaling")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return speedups

def exploit_digital_time_dilation():
    """Demonstrate practical applications of digital time dilation."""
    print("\n=== Exploiting Digital Time Dilation ===")
    
    # 1. Quantum Circuit Optimization
    print("\n1. Quantum Circuit Optimization")
    num_qubits = 5
    circuit_depth = 20
    
    # Create a complex circuit
    complex_circuit = create_complex_circuit(num_qubits, circuit_depth)
    
    # Initialize time dilation with high acceleration
    qtd = QuantumTimeDilation(
        num_qubits=num_qubits,
        num_streams=50,
        base_acceleration=50.0,
        predictive_depth=10,
        adaptive_rate=0.2,
        coherence_threshold=0.95
    )
    
    # Run accelerated computation
    print("Running accelerated computation...")
    start_time = time.time()
    results = qtd.accelerate_computation(complex_circuit, target_time=2.0)
    execution_time = time.time() - start_time
    
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Virtual time reached: {results['virtual_time_reached']:.4f}")
    print(f"Average performance: {results['average_performance']:.4f}")
    
    # 2. Quantum State Evolution Prediction
    print("\n2. Quantum State Evolution Prediction")
    
    # Get initial and final states
    initial_state = qtd.streams[0].quantum_state
    final_state = results['final_state']
    
    # Calculate state evolution metrics
    fidelity = np.abs(np.vdot(initial_state.data, final_state.data))**2
    print(f"Initial to final state fidelity: {fidelity:.4f}")
    
    # 3. Performance Visualization
    print("\n3. Performance Visualization")
    qtd.visualize_results(results)
    qtd.visualize_metrics()
    
    return results

def main():
    """Main function to run the demo."""
    print("=== Quantum Time Dilation Demo ===")
    print("This demo shows how quantum time dilation can accelerate quantum computations.")
    
    # Compare execution times
    print("\n=== Comparing Execution Times ===")
    speedup, results = compare_execution_times()
    
    # Demonstrate scaling
    print("\n=== Demonstrating Time Dilation Scaling ===")
    speedups = demonstrate_time_dilation_scaling()
    
    # Exploit digital time dilation
    print("\n=== Exploiting Digital Time Dilation ===")
    final_results = exploit_digital_time_dilation()
    
    print("\n=== Demo Completed ===")
    print(f"Maximum speedup achieved: {np.max(speedups):.2f}x")
    print(f"Average performance: {final_results['average_performance']:.4f}")
    print(f"Average coherence: {final_results['average_coherence']:.4f}")

if __name__ == "__main__":
    main() 