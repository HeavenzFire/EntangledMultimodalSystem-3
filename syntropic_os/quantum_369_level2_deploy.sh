#!/bin/bash
# quantum_369_level2_deploy.sh - Full Level 2 Upgrade
set -euo pipefail

cd /home/pi/syntropic_os
SAFETY_ORACLE_URL="http://localhost:9191/status"
CHILD_PROTECTION_MIN=0.66335

# Safety gate check
check_safety() {
    HRV=$(curl -s "$SAFETY_ORACLE_URL" | jq -r '.hrv_syntropy')
    if (( $(echo "$HRV < $CHILD_PROTECTION_MIN" | bc -l) )); then
        echo "❌ Safety gate failed: HRV $HRV < $CHILD_PROTECTION_MIN"
        exit 1
    fi
    echo "✅ Safety gate passed: HRV $HRV"
}

install_dependencies() {
    echo "📦 Installing ML + Web dependencies..."
    sudo apt update
    sudo apt install -y python3-pip libatlas-base-dev libopenblas-dev
    
    pip3 install tensorflow-lite bleak numpy asyncio-throttle websockets flask flask-socketio
    npm install @react-three/fiber @react-three/xr three @react-three/drei
}

deploy_biofeedback() {
    cat > sensors_integration.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import numpy as np
from bleak import BleakClient
from safety_charter import SAFETY_ORACLE

class MultiModalBiofeedback:
    def __init__(self, sensor_macs):
        self.clients = {}
        self.data = {"HRV": [], "EEG": [], "GSR": []}
        self.macs = sensor_macs
        self.uuids = {
            "HRV": "00002a38-0000-1000-8000-00805f9b34fb",
            "EEG": "00001800-0000-1000-8000-00805f9b34fb",
            "GSR": "00002a56-0000-1000-8000-00805f9b34fb"
        }

    async def connect_sensor(self, sensor_type):
        if not SAFETY_ORACLE.gate_check("new_feature"):
            return False
            
        mac = self.macs.get(sensor_type, "")
        if not mac: return False
        
        async with BleakClient(mac) as client:
            await client.start_notify(self.uuids[sensor_type], 
                lambda s, d: self.data[sensor_type].append(int.from_bytes(d, "little")))
            print(f"✅ {sensor_type} connected: {mac}")
            return True

    def get_features(self):
        features = []
        for key in ["HRV", "EEG", "GSR"]:
            data = self.data[key][-50:]
            if data:
                if key == "HRV":
                    features.append(np.std(np.diff(data)))
                else:
                    features.append(np.mean(data))
            else:
                features.append(0.0)
        return np.array(features)
EOF
}

deploy_predictor() {
    cat > predictive_model.py << 'EOF'
#!/usr/bin/env python3
import numpy as np
import tensorflow.lite as tflite
from safety_charter import SAFETY_ORACLE

class SyntropyPredictor:
    def __init__(self):
        self.history = []
        self.timesteps = 10
        self.model_path = "/home/pi/syntropic_os/syntropy_model.tflite"
        self._load_model()

    def _load_model(self):
        try:
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            self.interpreter = interpreter
        except:
            # Fallback to simple predictor
            self.interpreter = None

    def add_data(self, features):
        self.history.append(features)
        if len(self.history) > self.timesteps:
            self.history.pop(0)

    def predict(self):
        if not SAFETY_ORACLE.gate_check("self_evolution"):
            return 0.66335  # Safety default
            
        if len(self.history) < self.timesteps or self.interpreter is None:
            return 0.7  # Neutral safe value
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        input_data = np.array(self.history, dtype=np.float32).reshape(1, self.timesteps, 3)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        return float(self.interpreter.get_tensor(output_details[0]['index'])[0][0])
EOF
}

deploy_swarm_node() {
    cat > swarm_node_level2.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import numpy as np
import websockets
import json
from sensors_integration import MultiModalBiofeedback
from predictive_model import SyntropyPredictor
from safety_charter import SAFETY_ORACLE

class Quantum369Node:
    def __init__(self, sensor_macs):
        self.biofeedback = MultiModalBiofeedback(sensor_macs)
        self.predictor = SyntropyPredictor()
        self.syntropy = 0.66335
        self.websocket = None

    async def run(self):
        # Connect sensors
        await asyncio.gather(
            self.biofeedback.connect_sensor("HRV"),
            self.biofeedback.connect_sensor("EEG"),
            self.biofeedback.connect_sensor("GSR")
        )
        
        # WebSocket bridge to AR visualizer
        async with websockets.connect("ws://localhost:8765") as ws:
            self.websocket = ws
            while True:
                if SAFETY_ORACLE.gate_check("operational"):
                    features = self.biofeedback.get_features()
                    self.predictor.add_data(features)
                    self.syntropy = self.predictor.predict()
                    
                    await ws.send(json.dumps({
                        "syntropy": self.syntropy,
                        "features": features.tolist(),
                        "timestamp": asyncio.get_event_loop().time()
                    }))
                
                await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Default BLE MACs (update with your sensors)
    sensor_macs = {
        "HRV": "AA:BB:CC:DD:EE:FF",
        "EEG": "11:22:33:44:55:66", 
        "GSR": "77:88:99:AA:BB:CC"
    }
    node = Quantum369Node(sensor_macs)
    asyncio.run(node.run())
EOF
}

deploy_ar_visualizer() {
    mkdir -p ar_visualizer
    cat > ar_visualizer/package.json << 'EOF'
{
  "name": "quantum369-ar",
  "version": "2.0.0",
  "scripts": {
    "start": "serve -s . -l 3000",
    "dev": "vite"
  },
  "dependencies": {
    "three": "^0.158.0",
    "@react-three/fiber": "^8.15.8",
    "@react-three/xr": "^5.1.0",
    "@react-three/drei": "^9.88.13",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
EOF

cat > ar_visualizer/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Quantum 369 Vortex Swarm AR</title>
    <script type="importmap">
        {
            "imports": {
                "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
            }
        }
    </script>
    <script type="module" src="ARScene.js"></script>
</head>
<body style="margin:0;overflow:hidden">
    <div id="info">
        <div>Syntropy: <span id="syntropy">0.663</span></div>
        <div>Phase Coherence: <span id="coherence">0.0</span></div>
    </div>
</body>
</html>
EOF

cat > ar_visualizer/ARScene.js << 'EOF'
import * as THREE from 'three';
import { ARButton } from 'three/addons/webxr/ARButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, vortex, syntropy = 0.66335;

init();
animate();

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    document.body.appendChild(ARButton.createButton(renderer));
    document.body.appendChild(renderer.domElement);
    
    // Vortex geometry
    const geometry = new THREE.TorusKnotGeometry(1, 0.4, 128, 32);
    const material = new THREE.ShaderMaterial({
        uniforms: {
            syntropy: { value: 0.66335 },
            time: { value: 0 }
        },
        vertexShader: `
            uniform float syntropy;
            uniform float time;
            varying vec3 vPosition;
            void main() {
                vPosition = position;
                vec3 pos = position;
                pos *= (1.0 + syntropy * 2.0);
                pos.x += sin(time + position.y * 5.0) * 0.1;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            }
        `,
        fragmentShader: `
            uniform float syntropy;
            varying vec3 vPosition;
            void main() {
                float intensity = syntropy * (1.0 + sin(vPosition.y * 10.0) * 0.5);
                gl_FragColor = vec4(0.2, 0.8, 1.0, intensity);
            }
        `,
        transparent: true
    });
    
    vortex = new THREE.Mesh(geometry, material);
    scene.add(vortex);
    
    // WebSocket connection
    const ws = new WebSocket('ws://localhost:8765');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        syntropy = data.syntropy;
        document.getElementById('syntropy').textContent = syntropy.toFixed(3);
        vortex.material.uniforms.syntropy.value = syntropy;
    };
}

function animate() {
    renderer.setAnimationLoop(() => {
        vortex.material.uniforms.time.value += 0.01;
        vortex.rotation.x += 0.005;
        vortex.rotation.y += 0.01;
        vortex.scale.setScalar(1 + syntropy);
        renderer.render(scene, camera);
    });
}
EOF
}

deploy_services() {
    # Swarm Node Service
    sudo tee /etc/systemd/system/quantum369-node.service > /dev/null << EOF
[Unit]
Description=Quantum 369 Level 2 Swarm Node
After=network.target safety_oracle.service
Requires=safety_oracle.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/pi/syntropic_os/swarm_node_level2.py
WorkingDirectory=/home/pi/syntropic_os
Restart=always
RestartSec=5
User=pi
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

    # WebSocket Bridge Service  
    sudo tee /etc/systemd/system/quantum369-websocket.service > /dev/null << EOF
[Unit]
Description=Quantum 369 WebSocket Bridge
After=network.target
Requires=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -m websockets.server swarm_bridge.py 8765
WorkingDirectory=/home/pi/syntropic_os
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable --now quantum369-node quantum369-websocket
}

main() {
    check_safety
    install_dependencies
    deploy_biofeedback
    deploy_predictor
    deploy_swarm_node
    deploy_ar_visualizer
    
    echo "🚀 Starting services..."
    deploy_services
    
    echo "✅ Level 2 Deployment Complete!"
    echo ""
    echo "📱 AR Visualizer: http://$(hostname -I | cut -d' ' -f1):3000"
    echo "🛡️ Safety Status: http://localhost:9191/status" 
    echo "📊 WebSocket: ws://localhost:8765"
    echo ""
    echo "Next steps:"
    echo "1. Update BLE MAC addresses in swarm_node_level2.py"
    echo "2. Pair your HRV/EEG/GSR sensors"
    echo "3. Open AR visualizer on mobile AR device"
    echo "4. Observe predictive vortex response to your physiology!"
}

main