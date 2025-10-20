import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { BrainCircuit, Dna, Atom, Activity, Zap, Power, Settings2 } from 'lucide-react'
import { quantumWebSocket, QuantumState } from './services/quantumWebSocket'

interface FrequencyData {
    freq: number
    amplitude: number
}

interface HealingState {
    entanglement_ratio: number
    tumor_resonance: number
    healthy_resonance: number
    applied_frequencies: number[]
}

export const BioQuantumGrid: React.FC = () => {
    const [healingState, setHealingState] = useState<HealingState>({
        entanglement_ratio: 0,
        tumor_resonance: 0,
        healthy_resonance: 0,
        applied_frequencies: []
    })

    const [frequencyData, setFrequencyData] = useState<FrequencyData[]>([])
    const [isActive, setIsActive] = useState(false)
    const [quantumState, setQuantumState] = useState<QuantumState | null>(null)

    useEffect(() => {
        const subscription = quantumWebSocket.getState().subscribe(state => {
            setQuantumState(state)
            if (state) {
                setFrequencyData(state.resonance)
                setHealingState({
                    entanglement_ratio: state.status.entanglement_ratio,
                    tumor_resonance: state.status.tumor_resonance,
                    healthy_resonance: state.status.healthy_resonance,
                    applied_frequencies: state.field.filter(f => f.active).map(f => f.freq)
                })
            }
        })

        return () => subscription.unsubscribe()
    }, [])

    const handleFrequencyAdjust = (freq: number, amplitude: number) => {
        quantumWebSocket.adjustFrequency(freq, amplitude)
    }

    const handleMultiverseToggle = () => {
        quantumWebSocket.toggleMultiverse(!quantumState?.status.multiverse_mode)
    }

    const handleActivate = () => {
        setIsActive(!isActive)
        if (!isActive) {
            quantumWebSocket.activateHealing()
        }
    }

    return (
        <div className="bg-slate-900 text-emerald-400 p-8 rounded-3xl shadow-2xl">
            <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-4">
                    <BrainCircuit className="w-12 h-12 animate-pulse" />
                    <h1 className="text-4xl font-bold">Khemetic Healing Matrix v12.21.36</h1>
                </div>
                <div className="flex items-center gap-4">
                    <button
                        onClick={handleActivate}
                        className={`p-2 rounded-full ${isActive ? 'bg-red-500' : 'bg-emerald-500'} transition-colors`}
                        title={isActive ? "Deactivate System" : "Activate System"}
                        aria-label={isActive ? "Deactivate System" : "Activate System"}
                    >
                        <Power className="w-6 h-6" />
                    </button>
                    <button
                        onClick={handleMultiverseToggle}
                        className="p-2 rounded-full bg-purple-500/20 hover:bg-purple-500/40 transition-colors"
                        title="Toggle Multiverse Mode"
                        aria-label="Toggle Multiverse Mode"
                    >
                        <Settings2 className="w-6 h-6" />
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {/* Genomic Resonance Tuner */}
                <div className="p-6 border-2 border-emerald-500 rounded-xl">
                    <div className="flex items-center gap-4 mb-4">
                        <Dna className="w-8 h-8" />
                        <h2 className="text-2xl">Genomic Resonance Tuner</h2>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={frequencyData}>
                            <CartesianGrid stroke="#444" />
                            <XAxis
                                dataKey="freq"
                                stroke="#6ee7b7"
                                tickFormatter={(value) => `${value}Hz`}
                            />
                            <YAxis
                                stroke="#6ee7b7"
                                domain={[-1, 1]}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                    border: '1px solid #10b981'
                                }}
                            />
                            <Line
                                type="monotone"
                                dataKey="amplitude"
                                stroke="#10b981"
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                    <div className="mt-4 space-y-2">
                        {frequencyData.map(({ freq, amplitude }) => (
                            <div key={freq} className="flex items-center gap-4">
                                <input
                                    type="range"
                                    min="-1"
                                    max="1"
                                    step="0.1"
                                    value={amplitude}
                                    onChange={(e) => handleFrequencyAdjust(freq, parseFloat(e.target.value))}
                                    className="w-full"
                                    aria-label={`Adjust frequency ${freq}Hz`}
                                    title={`Adjust frequency ${freq}Hz`}
                                />
                                <span className="text-sm">{freq}Hz</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Quantum Biophotonic Field */}
                <div className="p-6 border-2 border-purple-500 rounded-xl">
                    <div className="flex items-center gap-4 mb-4">
                        <Atom className="w-8 h-8 animate-spin" />
                        <h2 className="text-2xl">Quantum Biophotonic Field</h2>
                    </div>
                    <div className="space-y-4">
                        {quantumState?.field.map(({ freq, active }) => (
                            <div key={freq} className="flex items-center gap-4">
                                <div className={`h-8 w-8 ${active ? 'bg-purple-500/60' : 'bg-purple-500/20'} rounded-full animate-pulse`} />
                                <span className="text-lg">{freq} Hz</span>
                                <Activity className={`ml-auto ${active ? 'text-emerald-400' : 'text-slate-500'}`} />
                            </div>
                        ))}
                    </div>
                </div>

                {/* Entanglement Status */}
                <div className="p-6 border-2 border-blue-500 rounded-xl">
                    <div className="flex items-center gap-4 mb-4">
                        <Zap className="w-8 h-8" />
                        <h2 className="text-2xl">Entanglement Status</h2>
                    </div>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span>Entanglement Ratio:</span>
                            <span className="text-xl font-bold">
                                {quantumState?.status.entanglement_ratio.toFixed(3) ?? '0.000'}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span>Tumor Resonance:</span>
                            <span className="text-xl font-bold">
                                {quantumState?.status.tumor_resonance ?? '0'}
                            </span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span>Healthy Resonance:</span>
                            <span className="text-xl font-bold">
                                {quantumState?.status.healthy_resonance ?? '0'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Status Bar */}
            <div className="mt-8 p-4 bg-slate-800 rounded-lg">
                <div className="flex items-center justify-between">
                    <span className={`${isActive ? 'text-emerald-400' : 'text-red-400'}`}>
                        System Status: {isActive ? 'Active' : 'Inactive'}
                    </span>
                    <span className="text-purple-400">
                        Multiverse Mode: {quantumState?.status.multiverse_mode ? 'Enabled' : 'Disabled'}
                    </span>
                    <span className="text-blue-400">
                        Quantum Entanglement: {quantumState?.status.quantum_stability ?? 'Unknown'}
                    </span>
                </div>
            </div>
        </div>
    )
} 