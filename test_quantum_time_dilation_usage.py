#!/usr/bin/env python3
"""
Test and validate the Enhanced Quantum Time Dilation system
and use it to advance quantum states.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from enhanced_quantum_time_dilation import QuantumTimeDilation

def create_test_circuit(num_qubits=3, depth=5):
    """Create a test quantum circuit with random gates."""
    qc = QuantumCircuit(num_qubits)
    
    # Apply Hadamard gates to create superposition
    qc.h(range(num_qubits))
    
    # Apply random gates for depth layers
    for _ in range(depth):
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
        
        # Apply some two-qubit gates
        for i in range(num_qubits-1):
            if np.random.random() > 0.5:
                qc.cx(i, i+1)
    
    # Measure all qubits
    qc.measure_all()
    
    return qc

def visualize_quantum_state(state_vector, title="Quantum State"):
    """Visualize a quantum state as a bar chart."""
    # Convert complex state vector to probabilities
    probabilities = np.abs(state_vector.data)**2
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(probabilities)), probabilities)
    plt.title(title)
    plt.xlabel("State Index")
    plt.ylabel("Probability")
    plt.xticks(range(len(probabilities)), [f"|{format(i, f'0{int(np.log2(len(probabilities)))}b')}>" for i in range(len(probabilities))])
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    num_qubits = 3
    num_streams = 10  # Reduced for testing
    target_time = 1.0
    
    print(f"Initializing Enhanced Quantum Time Dilation with {num_qubits} qubits and {num_streams} streams...")
    
    # Initialize the Quantum Time Dilation system
    qtd = QuantumTimeDilation(
        num_qubits=num_qubits,
        num_streams=num_streams,
        base_acceleration=1.0,  # Reduced for testing
        predictive_depth=5,
        adaptive_rate=0.1,
        coherence_threshold=0.95
    )
    
    # Create a test quantum circuit
    print("Creating test quantum circuit...")
    test_circuit = create_test_circuit(num_qubits=num_qubits, depth=3)
    print(test_circuit)
    
    # Get initial state of the first stream
    initial_state = qtd.streams[0].quantum_state
    print(f"Initial quantum state of stream 0: {initial_state}")
    visualize_quantum_state(initial_state, "Initial Quantum State")
    
    # Accelerate computation
    print(f"Accelerating computation to target time {target_time}...")
    results = qtd.accelerate_computation(test_circuit, target_time)
    
    # Print results summary
    print("\nResults Summary:")
    print(f"Virtual time reached: {results['virtual_time_reached']:.4f}")
    print(f"Average performance: {results['average_performance']:.4f}")
    print(f"Average coherence: {results['average_coherence']:.4f}")
    print(f"Number of predictions: {results['num_predictions']}")
    
    # Visualize final states
    print("\nVisualizing final quantum states...")
    for i, stream in enumerate(qtd.streams[:3]):  # Show first 3 streams
        visualize_quantum_state(stream.quantum_state, f"Final Quantum State - Stream {i}")
    
    # Visualize performance metrics
    print("\nVisualizing performance metrics...")
    qtd.visualize_metrics()
    
    # Visualize results
    print("\nVisualizing results...")
    qtd.visualize_results(results)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 