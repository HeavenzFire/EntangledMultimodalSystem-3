import { BehaviorSubject } from 'rxjs'

export interface QuantumState {
    resonance: {
        freq: number
        amplitude: number
    }[]
    field: {
        freq: number
        active: boolean
    }[]
    status: {
        entanglement_ratio: number
        tumor_resonance: number
        healthy_resonance: number
        multiverse_mode: boolean
        quantum_stability: string
    }
}

class QuantumWebSocket {
    private static instance: QuantumWebSocket
    private ws: WebSocket | null = null
    private state = new BehaviorSubject<QuantumState | null>(null)
    
    private constructor() {
        this.connect()
    }
    
    public static getInstance(): QuantumWebSocket {
        if (!QuantumWebSocket.instance) {
            QuantumWebSocket.instance = new QuantumWebSocket()
        }
        return QuantumWebSocket.instance
    }
    
    private connect() {
        this.ws = new WebSocket('ws://localhost:8080/quantum')
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            this.state.next(data)
        }
        
        this.ws.onclose = () => {
            setTimeout(() => this.connect(), 1000)
        }
    }
    
    public getState() {
        return this.state.asObservable()
    }
    
    public sendCommand(command: string, data: any) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command, data }))
        }
    }
    
    public adjustFrequency(freq: number, amplitude: number) {
        this.sendCommand('adjust_frequency', { freq, amplitude })
    }
    
    public toggleMultiverse(mode: boolean) {
        this.sendCommand('toggle_multiverse', { mode })
    }
    
    public activateHealing() {
        this.sendCommand('activate_healing', {})
    }
}

export const quantumWebSocket = QuantumWebSocket.getInstance() 