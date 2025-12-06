import express from 'express';
import * as http from 'http';
import * as WebSocket from 'ws';
import hrvRouter from './api/hrv_endpoint';

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(express.json());
app.use('/api', hrvRouter);

// Serve static files
app.use(express.static('src/biofeedback'));

// WebSocket for real-time HRV updates
wss.on('connection', (ws: WebSocket) => {
    console.log('Client connected to WebSocket');

    ws.on('message', (message: string) => {
        try {
            const data = JSON.parse(message);
            // Broadcast to all clients
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(JSON.stringify(data));
                }
            });
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected from WebSocket');
    });
});

// Start server
const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
    console.log(`Biofeedback server running on port ${PORT}`);
    console.log(`HRV visualization: http://localhost:${PORT}/hrv_visualization.html`);
    console.log(`API endpoints:`);
    console.log(`  POST /api/hrv - Receive HRV data`);
    console.log(`  GET /api/hrv - Get latest HRV data`);
    console.log(`  GET /api/status - Get system status`);
});

export default app;