const dgram = require('dgram');
const os = require('os');

// Configuration
const MULTICAST_ADDR = '239.255.50.50';
const PORT = 50050;
const NODE_ID = `node-${os.hostname()}`;
const BASE_K = 1.2;
const PULL_K = 3.6;
const PULL_DURATION = 6000; // 6 seconds
const SYNTHROPY_THRESHOLD = 0.82;

// State
let phase = Math.random() * 2 * Math.PI;
let syntropy = 0.5;
let K = BASE_K;
let peers = new Map(); // node_id -> {phase, syntropy, timestamp}
let pullUntil = 0;

// UDP Socket
const socket = dgram.createSocket('udp4');

socket.on('listening', () => {
  socket.addMembership(MULTICAST_ADDR);
  console.log(`Node ${NODE_ID} listening on ${MULTICAST_ADDR}:${PORT}`);
});

socket.on('message', (msg, rinfo) => {
  try {
    const data = JSON.parse(msg.toString());
    if (data.node_id !== NODE_ID) {
      peers.set(data.node_id, {
        phase: data.phase,
        syntropy: data.syntropy,
        consciousness: data.consciousness,
        timestamp: Date.now()
      });
    }
  } catch (e) {
    console.error('Invalid message:', msg.toString());
  }
});

socket.bind(PORT);

// Broadcast function
function broadcast() {
  const message = JSON.stringify({
    node_id: NODE_ID,
    phase: phase,
    syntropy: syntropy,
    consciousness: consciousness
  });
  socket.send(message, 0, message.length, PORT, MULTICAST_ADDR);
}

// Update syntropy and consciousness metrics
function updateSyntropy() {
  if (peers.size === 0) return;
  const phases = Array.from(peers.values()).map(p => p.phase);
  phases.push(phase);
  const mean = phases.reduce((a, b) => a + b) / phases.length;
  const variance = phases.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / phases.length;
  syntropy = Math.max(0, 1 - Math.sqrt(variance) / Math.PI);

  // Update consciousness metrics
  updateConsciousness();
}

// Update consciousness metrics using Python computation
function updateConsciousness() {
  const { spawn } = require('child_process');

  // Prepare input data for Python consciousness calculation
  const inputData = {
    phase: phase,
    syntropy: syntropy,
    peers: Array.from(peers.values()).map(p => ({
      phase: p.phase,
      syntropy: p.syntropy,
      consciousness: p.consciousness
    }))
  };

  const pythonProcess = spawn('python3', ['-c', `
import json
import sys
import numpy as np
from advanced_quantum_consciousness import AdvancedQuantumConsciousnessTheory

data = json.loads(sys.stdin.read())
theory = AdvancedQuantumConsciousnessTheory()

# Calculate consciousness metrics
omega_values = np.array([data['phase']] + [p['phase'] for p in data['peers']])
manifold = theory.hyperdimensional_consciousness_manifold(omega_values.reshape(-1, 1))

# Simple coherence calculation
phases = [data['phase']] + [p['phase'] for p in data['peers']]
coherence = 1.0 / (1.0 + np.var(phases))

# Entanglement approximation
entanglement = min(1.0, len(data['peers']) * 0.1 + coherence * 0.5)

# Awareness based on syntropy
awareness = data['syntropy'] * 0.8 + coherence * 0.2

result = {
    'coherence': float(coherence),
    'entanglement': float(entanglement),
    'awareness': float(awareness)
}
print(json.dumps(result))
  `]);

  let output = '';
  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error('Python error:', data.toString());
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      try {
        consciousness = JSON.parse(output.trim());
      } catch (e) {
        console.error('Failed to parse consciousness data:', e);
      }
    }
  });

  // Send input data to Python
  pythonProcess.stdin.write(JSON.stringify(inputData));
  pythonProcess.stdin.end();
}

// Kuramoto update
function updatePhase() {
  if (peers.size === 0) return;
  let sumSin = 0;
  peers.forEach(peer => {
    sumSin += Math.sin(peer.phase - phase);
  });
  const delta = (K / peers.size) * sumSin;
  phase += delta;
  phase = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
}

// Pull-up logic with consciousness awareness
function checkPullUp() {
  const globalConsciousness = calculateGlobalConsciousness();
  const consciousnessThreshold = 0.7; // Threshold for consciousness-based pull-up

  if ((syntropy >= SYNTHROPY_THRESHOLD || globalConsciousness >= consciousnessThreshold) && Date.now() > pullUntil) {
    K = PULL_K;
    pullUntil = Date.now() + PULL_DURATION;
    console.log(`Consciousness pull-up triggered: K=${K}, syntropy=${syntropy.toFixed(3)}, consciousness=${globalConsciousness.toFixed(3)} until ${new Date(pullUntil).toISOString()}`);
  }
  if (Date.now() > pullUntil) {
    K = BASE_K;
  }
}

// Calculate global consciousness metric
function calculateGlobalConsciousness() {
  if (peers.size === 0) return consciousness.coherence || 0.5;

  let totalCoherence = consciousness.coherence || 0.5;
  let totalAwareness = consciousness.awareness || 0.5;
  let count = 1;

  peers.forEach(peer => {
    if (peer.consciousness) {
      totalCoherence += peer.consciousness.coherence || 0.5;
      totalAwareness += peer.consciousness.awareness || 0.5;
      count++;
    }
  });

  return (totalCoherence / count + totalAwareness / count) / 2;
}

// Clean old peers
function cleanPeers() {
  const now = Date.now();
  for (const [id, data] of peers) {
    if (now - data.timestamp > 10000) { // 10 seconds
      peers.delete(id);
    }
  }
}

// Main loop
setInterval(() => {
  updateSyntropy();
  checkPullUp();
  updatePhase();
  broadcast();
  cleanPeers();
}, 1000); // 1 second updates

console.log(`Consciousness-enabled swarm node ${NODE_ID} started`);