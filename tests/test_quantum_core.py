import pytest
import numpy as np
from qiskit import QuantumCircuit
from core.quantum.qubit_control import TransmonQubit, PhotonicQubit, TopologicalQubit, QubitControlSystem
from core.quantum.error_correction.surface_code import SurfaceCode, SurfaceCodeParameters
from core.quantum.algorithms.quantum_algorithms import (
    GroverFractalSearch,
    QuantumBoltzmannMachine,
    VQERadiationModel
)

def test_transmon_qubit():
    """Test TransmonQubit implementation."""
    qubit = TransmonQubit(frequency=5.0, t1=150, t2_star=80, snr=15)
    
    # Test valid frequency range
    assert 4.5 <= qubit.frequency <= 5.5
    
    # Test flux bias application
    new_freq = qubit.apply_flux_bias(0.05)
    assert 4.5 <= new_freq <= 5.5
    
    # Test invalid flux bias
    with pytest.raises(ValueError):
        qubit.apply_flux_bias(0.2)

def test_photonic_qubit():
    """Test PhotonicQubit implementation."""
    qubit = PhotonicQubit(wavelength=1550, pulse_width=25, detection_efficiency=95)
    
    assert qubit.wavelength == 1550
    assert qubit.pulse_width == 25
    assert 0 <= qubit.detection_efficiency <= 100
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        PhotonicQubit(wavelength=1300, pulse_width=25, detection_efficiency=95)

def test_surface_code():
    """Test SurfaceCode implementation."""
    params = SurfaceCodeParameters(
        distance=7,
        physical_qubits_per_logical=49,
        syndrome_extraction_cycle=100e-9,
        stabilizer_measurement_time=450e-9
    )
    code = SurfaceCode(params)
    
    # Test logical qubit creation
    circuit = code.create_logical_qubit(0)
    assert isinstance(circuit, QuantumCircuit)
    assert len(circuit.qubits) == params.physical_qubits_per_logical + params.physical_qubits_per_logical // 2
    
    # Test lattice surgery
    circuit2 = code.create_logical_qubit(1)
    merged_circuit = code.perform_lattice_surgery(circuit, circuit2)
    assert isinstance(merged_circuit, QuantumCircuit)

def test_grover_fractal_search():
    """Test GroverFractalSearch implementation."""
    grover = GroverFractalSearch()
    
    # Test oracle creation
    oracle = grover.create_oracle("F[+F]F[-F]F")
    assert isinstance(oracle, QuantumCircuit)
    assert len(oracle.qubits) == grover.num_qubits
    
    # Test diffusion operator
    diffusion = grover.create_diffusion_operator()
    assert isinstance(diffusion, QuantumCircuit)
    assert len(diffusion.qubits) == 8

def test_quantum_boltzmann_machine():
    """Test QuantumBoltzmannMachine implementation."""
    qbm = QuantumBoltzmannMachine()
    
    # Test circuit creation
    circuit = qbm.create_qbm_circuit()
    assert isinstance(circuit, QuantumCircuit)
    assert len(circuit.qubits) == qbm.num_visible + qbm.num_hidden
    
    # Test parallel tempering
    energies = qbm.parallel_tempering(circuit)
    assert len(energies) == qbm.num_replicas
    assert all(isinstance(e, float) for e in energies)

def test_vqe_radiation_model():
    """Test VQERadiationModel implementation."""
    vqe = VQERadiationModel()
    
    # Test ansatz creation
    ansatz = vqe.create_ansatz(4)
    assert isinstance(ansatz, QuantumCircuit)
    
    # Test Hamiltonian creation
    alpha = [1.0, 2.0, 3.0, 4.0]
    J = 0.5
    hamiltonian = vqe.create_hamiltonian(alpha, J)
    assert hamiltonian.num_qubits == len(alpha)
    
    # Test parameter optimization
    eigenvalue, params = vqe.optimize_parameters(ansatz, hamiltonian)
    assert isinstance(eigenvalue, float)
    assert isinstance(params, np.ndarray)

def test_qubit_control_system():
    """Test integrated QubitControlSystem."""
    control_system = QubitControlSystem()
    
    # Test circuit creation
    circuit = control_system.create_quantum_circuit(4)
    assert isinstance(circuit, QuantumCircuit)
    assert len(circuit.qubits) == 4
    
    # Test measurement
    result = control_system.measure_qubit(circuit, 0)
    assert isinstance(result, float) 