from typing import Tuple, Optional
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class HybridKeyExchange:
    """Implements a hybrid quantum-classical key exchange protocol."""
    
    def __init__(self, quantum_bits: int = 4):
        """Initialize the hybrid key exchange system."""
        self.quantum_bits = quantum_bits
        self.backend = Aer.get_backend('qasm_simulator')
        self.classical_curve = ec.SECP384R1()
        
    def generate_classical_keypair(self) -> Tuple[ec.EllipticCurvePrivateKey, 
                                                ec.EllipticCurvePublicKey]:
        """Generate a classical ECDH key pair."""
        private_key = ec.generate_private_key(self.classical_curve)
        public_key = private_key.public_key()
        return private_key, public_key
        
    def _create_bb84_circuit(self, bits: str, bases: str) -> QuantumCircuit:
        """Create a BB84 quantum circuit for key distribution."""
        qr = QuantumRegister(len(bits), 'q')
        cr = ClassicalRegister(len(bits), 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Prepare qubits according to bits and bases
        for i, (bit, basis) in enumerate(zip(bits, bases)):
            if bit == '1':
                circuit.x(qr[i])
            if basis == '1':  # Hadamard basis
                circuit.h(qr[i])
                
        # Measure in computational basis
        circuit.measure(qr, cr)
        return circuit
        
    def generate_quantum_key(self, length: int) -> Tuple[str, str, str]:
        """Generate quantum key using BB84-like protocol."""
        # Generate random bits and bases
        bits = ''.join(np.random.choice(['0', '1']) for _ in range(length))
        bases = ''.join(np.random.choice(['0', '1']) for _ in range(length))
        
        # Create and execute quantum circuit
        circuit = self._create_bb84_circuit(bits, bases)
        job = execute(circuit, self.backend, shots=1)
        result = job.result().get_counts(circuit)
        measured = list(result.keys())[0]  # Get the measured bitstring
        
        return bits, bases, measured
        
    def combine_keys(self, classical_shared: bytes, quantum_key: str) -> bytes:
        """Combine classical and quantum keys using HKDF."""
        # Use HKDF to derive the final key
        hkdf = HKDF(
            algorithm=hashes.SHA384(),
            length=32,
            salt=None,
            info=b'hybrid_key_exchange'
        )
        
        # Combine classical and quantum materials
        combined = classical_shared + quantum_key.encode()
        return hkdf.derive(combined)
        
    def perform_key_exchange(self) -> Tuple[bytes, float]:
        """Perform complete hybrid key exchange."""
        # Classical key exchange
        alice_private, alice_public = self.generate_classical_keypair()
        bob_private, bob_public = self.generate_classical_keypair()
        
        # Generate classical shared secret
        alice_shared = alice_private.exchange(ec.ECDH(), bob_public)
        bob_shared = bob_private.exchange(ec.ECDH(), alice_public)
        
        # Quantum key distribution
        alice_bits, alice_bases, _ = self.generate_quantum_key(self.quantum_bits)
        bob_bits, bob_bases, measured = self.generate_quantum_key(self.quantum_bits)
        
        # Sift quantum key
        quantum_key = ''
        matches = 0
        for i, (a_base, b_base, meas) in enumerate(zip(alice_bases, bob_bases, measured)):
            if a_base == b_base:  # Same basis used
                quantum_key += meas
                if meas == alice_bits[i]:
                    matches += 1
                    
        # Calculate quantum bit error rate (QBER)
        qber = 1 - (matches / len(quantum_key)) if quantum_key else 1.0
        
        # Combine keys
        final_key = self.combine_keys(alice_shared, quantum_key)
        
        return final_key, qber
        
    def validate_exchange(self, key: bytes, qber: float, 
                         max_qber: float = 0.11) -> bool:
        """Validate the key exchange results."""
        # Check QBER is below threshold (typical BB84 threshold is 11%)
        if qber > max_qber:
            return False
            
        # Verify key length
        if len(key) != 32:  # Expected 256-bit key
            return False
            
        # Additional entropy tests could be added here
        
        return True 