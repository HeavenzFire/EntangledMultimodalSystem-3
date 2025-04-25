#!/usr/bin/env python3
"""
Example script demonstrating the usage of Quantum Time Dilation Framework.
This script shows how to:
1. Create complex quantum circuits
2. Apply time dilation acceleration
3. Visualize results and performance metrics
4. Compare with standard quantum execution
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from quantum_time_dilation import QuantumTimeDilation

def create_quantum_fourier_circuit(num_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """
    Create a quantum circuit implementing Quantum Fourier Transform.
    
    Args:
        num_qubits: Number of qubits in the circuit
        inverse: Whether to create inverse QFT
        
    Returns:
        QuantumCircuit implementing QFT
    """
    qc = QuantumCircuit(num_qubits)
    qft = QFT(num_qubits, inverse=inverse)
    qc.compose(qft, inplace=True)
    qc.measure_all()
    return qc

def create_entangled_state_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit generating highly entangled states.
    
    Args:
        num_qubits: Number of qubits in the circuit
        
    Returns:
        QuantumCircuit generating entangled states
    """
    qc = QuantumCircuit(num_qubits)
    
    # Create superposition
    qc.h(range(num_qubits))
    
    # Create entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.h(i + 1)
    
    qc.measure_all()
    return qc

def compare_execution_times(circuit: QuantumCircuit, 
                          num_streams: int = 1000,
                          target_time: float = 1.0) -> dict:
    """
    Compare execution times between standard and time-dilated quantum computation.
    
    Args:
        circuit: Quantum circuit to execute
        num_streams: Number of parallel streams for time dilation
        target_time: Target virtual time to reach
        
    Returns:
        Dictionary containing timing results
    """
    # Standard execution
    start_time = time.time()
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, simulator).result()
    standard_time = time.time() - start_time
    
    # Time-dilated execution
    qtd = QuantumTimeDilation(num_streams=num_streams)
    start_time = time.time()
    dilated_results = qtd.accelerate_computation(circuit, target_time)
    dilated_time = time.time() - start_time
    
    return {
        'standard_time': standard_time,
        'dilated_time': dilated_time,
        'speedup_factor': standard_time / dilated_time,
        'dilated_results': dilated_results
    }

def visualize_results(results: dict, circuit: QuantumCircuit):
    """
    Visualize execution results and performance metrics.
    
    Args:
        results: Dictionary containing execution results
        circuit: Original quantum circuit
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot execution times
    times = [results['standard_time'], results['dilated_time']]
    ax1.bar(['Standard', 'Time-Dilated'], times)
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    
    # Plot state variance
    variance = results['dilated_results']['state_variance']
    ax2.plot(variance)
    ax2.set_title('State Variance Across Streams')
    ax2.set_xlabel('State Component')
    ax2.set_ylabel('Variance')
    
    # Plot virtual time progression
    virtual_time = results['dilated_results']['virtual_time_reached']
    ax3.text(0.5, 0.5, f'Virtual Time: {virtual_time:.2f}',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax3.transAxes)
    ax3.set_title('Virtual Time Achievement')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run Quantum Time Dilation examples")
    parser.add_argument("--qubits", type=int, default=5, help="Number of qubits")
    parser.add_argument("--streams", type=int, default=1000, help="Number of parallel streams")
    parser.add_argument("--time", type=float, default=1.0, help="Target virtual time")
    parser.add_argument("--circuit", choices=['qft', 'entangled'], default='qft',
                      help="Type of quantum circuit to use")
    args = parser.parse_args()
    
    # Create quantum circuit
    if args.circuit == 'qft':
        circuit = create_quantum_fourier_circuit(args.qubits)
    else:
        circuit = create_entangled_state_circuit(args.qubits)
    
    print(f"\nRunning quantum computation with {args.qubits} qubits...")
    print(f"Circuit type: {args.circuit}")
    print(f"Number of streams: {args.streams}")
    print(f"Target virtual time: {args.time}")
    
    # Compare execution times
    results = compare_execution_times(circuit, args.streams, args.time)
    
    print("\nResults:")
    print(f"Standard execution time: {results['standard_time']:.3f} seconds")
    print(f"Time-dilated execution time: {results['dilated_time']:.3f} seconds")
    print(f"Speedup factor: {results['speedup_factor']:.2f}x")
    print(f"Virtual time reached: {results['dilated_results']['virtual_time_reached']:.2f}")
    print(f"Total predictions made: {results['dilated_results']['num_predictions']}")
    
    # Visualize results
    visualize_results(results, circuit)

if __name__ == "__main__":
    main() 