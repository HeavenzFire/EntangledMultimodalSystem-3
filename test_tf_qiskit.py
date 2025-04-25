import tensorflow as tf
import qiskit
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Qiskit version: {qiskit.__version__}")

# Test TensorFlow
print("\nTesting TensorFlow:")
x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[5, 6], [7, 8]])
z = tf.matmul(x, y)
print("Matrix multiplication result:")
print(z.numpy())

# Test Qiskit
print("\nTesting Qiskit:")
circuit = qiskit.QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
print("Quantum circuit:")
print(circuit) 