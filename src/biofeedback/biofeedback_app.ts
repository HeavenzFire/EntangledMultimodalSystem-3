import { HRVRing } from './hrv_ring';
import { MQTTClient, HRVData } from './mqtt_client';

export class BiofeedbackApp {
    private hrvRing: HRVRing;
    private mqttClient: MQTTClient;
    private canvas: HTMLCanvasElement;
    private statusElement: HTMLElement;

    constructor(canvas: HTMLCanvasElement, statusElement: HTMLElement) {
        this.canvas = canvas;
        this.statusElement = statusElement;

        // Initialize HRV Ring visualization
        this.hrvRing = new HRVRing(canvas);

        // Initialize MQTT client
        this.mqttClient = new MQTTClient('ws://localhost:9001', this.handleHRVData.bind(this));

        // Setup canvas resize
        window.addEventListener('resize', this.handleResize.bind(this));
        this.handleResize();

        this.updateStatus('Initializing biofeedback system...');
    }

    private handleHRVData(data: HRVData): void {
        console.log('Received HRV data:', data);

        // Update visualization
        this.hrvRing.updateHRV(data.rmssd, data.coherence);

        // Update status
        this.updateStatus(`HRV RMSSD: ${data.rmssd.toFixed(3)}, Coherence: ${(data.coherence * 100).toFixed(1)}%`);
    }

    private updateStatus(message: string): void {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
    }

    private handleResize(): void {
        const rect = this.canvas.getBoundingClientRect();
        this.hrvRing.resize(rect.width, rect.height);
    }

    dispose(): void {
        this.mqttClient.disconnect();
        this.hrvRing.dispose();
        window.removeEventListener('resize', this.handleResize);
    }
}