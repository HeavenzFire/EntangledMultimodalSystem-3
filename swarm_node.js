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
    syntropy: syntropy
  });
  socket.send(message, 0, message.length, PORT, MULTICAST_ADDR);
}

// Update syntropy (simple coherence measure)
function updateSyntropy() {
  if (peers.size === 0) return;
  const phases = Array.from(peers.values()).map(p => p.phase);
  phases.push(phase);
  const mean = phases.reduce((a, b) => a + b) / phases.length;
  const variance = phases.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / phases.length;
  syntropy = Math.max(0, 1 - Math.sqrt(variance) / Math.PI);
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

// Pull-up logic
function checkPullUp() {
  if (syntropy >= SYNTHROPY_THRESHOLD && Date.now() > pullUntil) {
    K = PULL_K;
    pullUntil = Date.now() + PULL_DURATION;
    console.log(`Pull-up triggered: K=${K} until ${new Date(pullUntil).toISOString()}`);
  }
  if (Date.now() > pullUntil) {
    K = BASE_K;
  }
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

console.log(`Swarm node ${NODE_ID} started`);