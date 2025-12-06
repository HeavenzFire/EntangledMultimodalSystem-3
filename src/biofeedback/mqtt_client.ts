import * as mqtt from 'mqtt';

export interface HRVData {
    rmssd: number;
    coherence: number;
    timestamp: number;
}

export class MQTTClient {
    private client: mqtt.MqttClient;
    private onHRVData: (data: HRVData) => void;

    constructor(brokerUrl: string, onHRVData: (data: HRVData) => void) {
        this.onHRVData = onHRVData;
        this.client = mqtt.connect(brokerUrl);

        this.client.on('connect', () => {
            console.log('Connected to MQTT broker');
            this.client.subscribe('home/sensor/hrv', (err) => {
                if (err) {
                    console.error('Failed to subscribe to HRV topic:', err);
                }
            });
        });

        this.client.on('message', (topic, message) => {
            if (topic === 'home/sensor/hrv') {
                try {
                    const data: HRVData = JSON.parse(message.toString());
                    this.onHRVData(data);
                } catch (error) {
                    console.error('Failed to parse HRV data:', error);
                }
            }
        });

        this.client.on('error', (error) => {
            console.error('MQTT connection error:', error);
        });
    }

    publish(topic: string, message: any): void {
        this.client.publish(topic, JSON.stringify(message));
    }

    disconnect(): void {
        this.client.end();
    }
}