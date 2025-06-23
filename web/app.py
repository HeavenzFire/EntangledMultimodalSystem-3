from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from src.quantum.security import QuantumSecurityFramework
from src.quantum.synthesis import QuantumSacredSynthesis
from src.quantum.geometry.entanglement_torus import QuantumEntanglementTorus

app = Flask(__name__)
CORS(app)

# Initialize quantum components
security_framework = QuantumSecurityFramework()
synthesis_system = QuantumSacredSynthesis()
entanglement_torus = QuantumEntanglementTorus()

@app.route('/api/quantum/security/encrypt', methods=['POST'])
def encrypt_data():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        encrypted_data, tag = security_framework.encrypt_data(data.encode())
        return jsonify({
            'encrypted_data': encrypted_data.hex(),
            'tag': tag.hex()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum/security/decrypt', methods=['POST'])
def decrypt_data():
    data = request.json
    if not all(k in data for k in ['encrypted_data', 'tag', 'iv']):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        decrypted = security_framework.decrypt_data(
            bytes.fromhex(data['encrypted_data']),
            bytes.fromhex(data['tag']),
            bytes.fromhex(data['iv'])
        )
        return jsonify({'decrypted_data': decrypted.decode()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum/synthesis/update', methods=['POST'])
def update_synthesis():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        result = synthesis_system.update_state(data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum/torus/harmonize', methods=['POST'])
def harmonize_field():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        result = entanglement_torus.harmonize_field(np.array(data))
        return jsonify({'result': result.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum/metrics', methods=['GET'])
def get_metrics():
    try:
        security_metrics = security_framework.get_security_metrics()
        synthesis_metrics = synthesis_system.get_metrics()
        torus_metrics = entanglement_torus.get_metrics()
        
        return jsonify({
            'security': security_metrics,
            'synthesis': synthesis_metrics,
            'torus': torus_metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 