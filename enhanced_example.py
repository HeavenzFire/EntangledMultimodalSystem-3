#!/usr/bin/env python3
"""
Enhanced Example of Quantum Time Dilation Framework
================================================
This script demonstrates the advanced features of the Quantum Time Dilation Framework:
1. Adaptive acceleration based on performance metrics
2. Coherence protection for quantum states
3. Visualization of quantum state evolution and acceleration distribution
4. Performance comparison with standard quantum execution
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.library import QFT
from quantum_time_dilation import QuantumTimeDilation

def create_complex_circuit(num_qubits, circuit_type="qft"):
    """
    Create a complex quantum circuit for testing.
    
    Args:
        num_qubits: Number of qubits in the circuit
        circuit_type: Type of circuit to create ("qft", "entangled", "random")
        
    Returns:
        QuantumCircuit: The created circuit
    """
    qc = QuantumCircuit(num_qubits)
    
    if circuit_type == "qft":
        # Quantum Fourier Transform
        qft = QFT(num_qubits)
        qc.compose(qft, inplace=True)
    elif circuit_type == "entangled":
        # Highly entangled state
        qc.h(range(num_qubits))
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.h(range(num_qubits))
    else:  # random
        # Random circuit with various gates
        for i in range(num_qubits):
            qc.h(i)
            if i < num_qubits - 1:
                qc.cx(i, i + 1)
        for i in range(num_qubits):
            qc.rz(np.random.random() * 2 * np.pi, i)
    
    qc.measure_all()
    return qc

def compare_standard_vs_enhanced(circuit, num_streams, target_time):
    """
    Compare standard quantum execution with enhanced time dilation.
    
    Args:
        circuit: Quantum circuit to execute
        num_streams: Number of parallel streams
        target_time: Target virtual time
        
    Returns:
        Dictionary with comparison results
    """
    # Standard execution
    start_time = time.time()
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, simulator).result()
    standard_time = time.time() - start_time
    standard_state = result.get_statevector()
    
    # Enhanced time dilation execution
    qtd = QuantumTimeDilation(
        num_streams=num_streams,
        adaptive_rate=0.1,
        coherence_threshold=0.95
    )
    
    start_time = time.time()
    dilated_results = qtd.accelerate_computation(circuit, target_time)
    dilated_time = time.time() - start_time
    
    return {
        'standard_time': standard_time,
        'dilated_time': dilated_time,
        'speedup_factor': standard_time / dilated_time,
        'standard_state': standard_state,
        'dilated_results': dilated_results
    }

def visualize_comparison(comparison_results, save_path=None):
    """
    Visualize comparison between standard and enhanced execution.
    
    Args:
        comparison_results: Results from compare_standard_vs_enhanced
        save_path: Optional path to save visualization
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Execution time comparison
    ax1 = fig.add_subplot(221)
    times = [comparison_results['standard_time'], comparison_results['dilated_time']]
    ax1.bar(['Standard', 'Time-Dilated'], times, color=['blue', 'green'])
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.text(0.5, 0.9, f"Speedup: {comparison_results['speedup_factor']:.2f}x", 
             horizontalalignment='center', transform=ax1.transAxes)
    
    # 2. State fidelity comparison
    ax2 = fig.add_subplot(222)
    standard_state = comparison_results['standard_state']
    dilated_state = comparison_results['dilated_results']['final_state']
    
    # Calculate fidelity
    fidelity = np.abs(np.vdot(standard_state, dilated_state))**2
    ax2.bar(['State Fidelity'], [fidelity], color='purple')
    ax2.set_ylim(0, 1)
    ax2.set_title('Quantum State Fidelity')
    ax2.text(0.5, 0.9, f"Fidelity: {fidelity:.4f}", 
             horizontalalignment='center', transform=ax2.transAxes)
    
    # 3. Performance metrics over time
    ax3 = fig.add_subplot(223)
    # Sample a few streams for visualization
    sample_size = min(5, len(comparison_results['dilated_results']['performance_metrics']))
    sample_indices = np.random.choice(
        len(comparison_results['dilated_results']['performance_metrics']), 
        sample_size, replace=False
    )
    
    for idx in sample_indices:
        metrics = comparison_results['dilated_results']['performance_metrics'][idx]
        time_points = np.linspace(0, 1, len(metrics))
        ax3.plot(time_points, metrics, label=f'Stream {idx}')
    
    ax3.set_title('Performance Metrics Over Time')
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Performance Metric')
    ax3.legend()
    
    # 4. Acceleration factor distribution
    ax4 = fig.add_subplot(224)
    accel_dist = comparison_results['dilated_results']['acceleration_distribution']
    ax4.hist([stream.acceleration_factor for stream in 
              QuantumTimeDilation(num_streams=1000).time_streams], 
             bins=30, alpha=0.7, color='orange')
    ax4.set_title('Acceleration Factor Distribution')
    ax4.set_xlabel('Acceleration Factor')
    ax4.set_ylabel('Frequency')
    ax4.text(0.5, 0.9, f"Mean: {accel_dist['mean']:.2e}", 
             horizontalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Enhanced Quantum Time Dilation Example")
    parser.add_argument("--qubits", type=int, default=5, help="Number of qubits")
    parser.add_argument("--streams", type=int, default=1000, help="Number of parallel streams")
    parser.add_argument("--time", type=float, default=1.0, help="Target virtual time")
    parser.add_argument("--circuit", choices=["qft", "entangled", "random"], default="qft",
                      help="Type of quantum circuit to use")
    parser.add_argument("--save", type=str, help="Path to save visualization")
    args = parser.parse_args()
    
    print(f"Creating {args.circuit} circuit with {args.qubits} qubits...")
    circuit = create_complex_circuit(args.qubits, args.circuit)
    
    print(f"Comparing standard vs. enhanced execution with {args.streams} streams...")
    comparison = compare_standard_vs_enhanced(circuit, args.streams, args.time)
    
    print("\nResults:")
    print(f"Standard execution time: {comparison['standard_time']:.3f} seconds")
    print(f"Time-dilated execution time: {comparison['dilated_time']:.3f} seconds")
    print(f"Speedup factor: {comparison['speedup_factor']:.2f}x")
    print(f"Average performance: {comparison['dilated_results']['average_performance']:.4f}")
    print(f"Acceleration distribution: {comparison['dilated_results']['acceleration_distribution']}")
    
    # Calculate fidelity
    fidelity = np.abs(np.vdot(comparison['standard_state'], 
                             comparison['dilated_results']['final_state']))**2
    print(f"Quantum state fidelity: {fidelity:.4f}")
    
    # Visualize results
    visualize_comparison(comparison, args.save)

if __name__ == "__main__":
    main() 