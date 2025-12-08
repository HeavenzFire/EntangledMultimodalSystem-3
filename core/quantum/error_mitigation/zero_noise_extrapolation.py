import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import NoiseModel
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit

class ZeroNoiseExtrapolator:
    """Implements Zero-Noise Extrapolation for quantum error mitigation."""
    
    def __init__(self, noise_scales: List[float] = None):
        """Initialize the extrapolator with noise scaling factors."""
        self.noise_scales = noise_scales or [1.0, 1.5, 2.0, 2.5]
        self.backend = Aer.get_backend('qasm_simulator')
        
    def _scale_circuit(self, circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
        """Scale the noise in a quantum circuit by repeating gates."""
        if scale_factor == 1.0:
            return circuit
            
        scaled_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        for instruction in circuit.data:
            # Repeat each gate operation scale_factor times
            gate = instruction[0]
            qubits = instruction[1]
            for _ in range(int(scale_factor)):
                scaled_circuit.append(gate, qubits)
                
        return scaled_circuit
        
    def _fit_model(self, scales: List[float], values: List[float], 
                   model: str = 'linear') -> Tuple[np.ndarray, float]:
        """Fit extrapolation model to the data."""
        if model == 'linear':
            fit_func = lambda x, a, b: a * x + b
        elif model == 'exponential':
            fit_func = lambda x, a, b, c: a * np.exp(b * x) + c
        else:
            raise ValueError(f"Unsupported model: {model}")
            
        popt, _ = curve_fit(fit_func, scales, values)
        zero_noise_value = fit_func(0, *popt)
        return popt, zero_noise_value
        
    def run_circuit_with_noise(self, circuit: QuantumCircuit, 
                             noise_model: Optional[NoiseModel] = None,
                             shots: int = 1024) -> dict:
        """Execute circuit with different noise scales and extrapolate to zero noise."""
        results = []
        
        for scale in self.noise_scales:
            scaled_circuit = self._scale_circuit(circuit, scale)
            job = execute(scaled_circuit, 
                         self.backend,
                         noise_model=noise_model,
                         shots=shots)
            counts = job.result().get_counts()
            
            # Calculate expectation value (assuming computational basis measurement)
            expectation = 0
            for bitstring, count in counts.items():
                # Convert bitstring to +1/-1 value
                value = 1 if bitstring.count('1') % 2 == 0 else -1
                expectation += value * count / shots
            results.append(expectation)
            
        # Fit models and extrapolate
        linear_params, linear_zero = self._fit_model(self.noise_scales, results, 'linear')
        exp_params, exp_zero = self._fit_model(self.noise_scales, results, 'exponential')
        
        return {
            'noise_scales': self.noise_scales,
            'measured_values': results,
            'linear_extrapolation': linear_zero,
            'exponential_extrapolation': exp_zero,
            'linear_params': linear_params,
            'exponential_params': exp_params
        }
        
    def validate_extrapolation(self, results: dict, tolerance: float = 0.1) -> bool:
        """Validate the extrapolation results."""
        linear_zero = results['linear_extrapolation']
        exp_zero = results['exponential_extrapolation']
        
        # Check if linear and exponential models agree within tolerance
        if abs(linear_zero - exp_zero) > tolerance:
            return False
            
        # Check if extrapolation is physically meaningful
        if abs(linear_zero) > 1.0 or abs(exp_zero) > 1.0:
            return False
            
        return True 