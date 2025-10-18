#!/usr/bin/env python3
"""
Example script demonstrating the usage of ShorAlgorithm for integer factorization.
This script shows how to:
1. Initialize the ShorAlgorithm class
2. Factor small numbers using different parameters
3. Visualize the quantum circuit
4. Handle errors and edge cases
"""

import argparse
import time
from typing import List, Optional
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.primitives import Sampler

from quantum_shor import ShorAlgorithm

def factor_number(N: int, a: int = 2, shots: int = 1000, verbose: bool = True) -> List[int]:
    """
    Factor a number using Shor's algorithm with progress reporting.
    
    Args:
        N: The number to factor
        a: Base for modular exponentiation
        shots: Number of quantum measurements
        verbose: Whether to print progress information
        
    Returns:
        List of factors found
    """
    if verbose:
        print(f"\nAttempting to factor N = {N} using base a = {a}")
        print(f"Using {shots} quantum measurements...")
    
    start_time = time.time()
    
    try:
        # Initialize Shor's algorithm
        shor = ShorAlgorithm()
        
        # Construct the circuit (without measurement for visualization)
        circuit = shor.construct_circuit(N, a, measurement=False)
        if verbose:
            print(f"\nCircuit depth: {circuit.depth()}")
            print(f"Total qubits: {circuit.num_qubits}")
            print(f"Total gates: {circuit.size()}")
        
        # Run the factorization
        factors = shor.factor(N, a, shots)
        
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\nFactorization completed in {elapsed_time:.2f} seconds")
            if factors:
                print(f"Factors found: {factors}")
            else:
                print("No factors found. Try increasing the number of shots.")
        
        return factors
        
    except ValueError as e:
        if verbose:
            print(f"Error: {str(e)}")
        return []
    except Exception as e:
        if verbose:
            print(f"Unexpected error: {str(e)}")
        return []

def visualize_circuit(N: int, a: int = 2):
    """
    Visualize the quantum circuit for Shor's algorithm.
    
    Args:
        N: The number to factor
        a: Base for modular exponentiation
    """
    shor = ShorAlgorithm()
    circuit = shor.construct_circuit(N, a, measurement=False)
    
    # Draw the circuit
    fig = circuit.draw(output='mpl')
    plt.title(f"Shor's Algorithm Circuit for N={N}, a={a}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run Shor's algorithm for integer factorization")
    parser.add_argument("N", type=int, help="Number to factor")
    parser.add_argument("--base", "-a", type=int, default=2, help="Base for modular exponentiation")
    parser.add_argument("--shots", "-s", type=int, default=1000, help="Number of quantum measurements")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the quantum circuit")
    args = parser.parse_args()
    
    # Factor the number
    factors = factor_number(args.N, args.base, args.shots)
    
    # Visualize the circuit if requested
    if args.visualize:
        visualize_circuit(args.N, args.base)

if __name__ == "__main__":
    main() 