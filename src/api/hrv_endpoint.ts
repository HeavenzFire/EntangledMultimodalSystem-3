import express from 'express';
import { WatcherImpl, run_simulation } from '../watcher_simulation';

const router = express.Router();

// In-memory storage for latest HRV data
let latestHRVData = {
    rmssd: 0.5,
    coherence: 0.5,
    timestamp: Date.now()
};

// POST endpoint to receive HRV data from sensors
router.post('/hrv', express.json(), (req, res) => {
    try {
        const { rmssd, coherence } = req.body;

        if (typeof rmssd !== 'number' || typeof coherence !== 'number') {
            return res.status(400).json({ error: 'Invalid HRV data format' });
        }

        latestHRVData = {
            rmssd,
            coherence,
            timestamp: Date.now()
        };

        console.log(`[HRV Endpoint] Received data: RMSSD=${rmssd.toFixed(3)}, Coherence=${(coherence * 100).toFixed(1)}%`);

        res.json({ status: 'ok', data: latestHRVData });
    } catch (error) {
        console.error('Error processing HRV data:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// GET endpoint to retrieve latest HRV data
router.get('/hrv', (req, res) => {
    res.json(latestHRVData);
});

// GET endpoint for watcher status integrated with HRV
router.get('/status', async (req, res) => {
    try {
        // Create a temporary watcher for status
        const watcher = new WatcherImpl("BiofeedbackWatcher", 1, {
            origin: "HRV Sensor",
            age: 0.001, // Very young, real-time
            resonance: latestHRVData.coherence,
            purpose: "Monitor biofeedback coherence"
        });

        watcher.scan_concepts(["child protection", "emotional clarity"], {});
        watcher.bind(528.0, "hrv_sensor"); // Healing frequency

        const speech = watcher.speak(latestHRVData.rmssd);

        const status = {
            timestamp: new Date().toISOString(),
            hrv_data: latestHRVData,
            watcher_status: speech,
            child_protection_ok: latestHRVData.coherence > 0.7,
            system_health: latestHRVData.coherence > 0.5 ? 'GOOD' : 'MONITORING'
        };

        res.json(status);
    } catch (error) {
        console.error('Error generating status:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

export default router;