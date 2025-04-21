import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt

class ResonantFrequencyProcessor:
    def __init__(self):
        """Initialize the resonant frequency processor."""
        try:
            self.sacred_frequencies = {
                'solfeggio': [174, 285, 396, 417, 528, 639, 741, 852, 963],
                'earth': [7.83, 14.3, 20.8, 27.3, 33.8],  # Schumann resonances
                'cosmic': [432, 528, 639, 741, 852, 963]  # Cosmic harmonics
            }
            self.harmonic_ratios = self._calculate_harmonic_ratios()
            logger.info("ResonantFrequencyProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ResonantFrequencyProcessor: {str(e)}")
            raise ModelError(f"Resonant frequency initialization failed: {str(e)}")

    def _calculate_harmonic_ratios(self):
        """Calculate harmonic ratios based on sacred frequencies."""
        ratios = {}
        for name, frequencies in self.sacred_frequencies.items():
            ratios[name] = []
            for i in range(len(frequencies)-1):
                ratio = frequencies[i+1] / frequencies[i]
                ratios[name].append(ratio)
        return ratios

    def analyze_frequencies(self, signal, sample_rate=44100):
        """Analyze frequencies in a signal."""
        try:
            # Perform FFT
            n = len(signal)
            yf = fft(signal)
            xf = fftfreq(n, 1/sample_rate)
            
            # Calculate power spectrum
            power_spectrum = np.abs(yf[:n//2])**2
            
            # Find dominant frequencies
            dominant_freqs = self._find_dominant_frequencies(xf[:n//2], power_spectrum)
            
            # Calculate harmonic relationships
            harmonics = self._analyze_harmonics(dominant_freqs)
            
            # Calculate resonance patterns
            resonance = self._calculate_resonance(dominant_freqs)
            
            return {
                "frequencies": xf[:n//2].tolist(),
                "power_spectrum": power_spectrum.tolist(),
                "dominant_frequencies": dominant_freqs,
                "harmonics": harmonics,
                "resonance_patterns": resonance
            }
        except Exception as e:
            logger.error(f"Frequency analysis failed: {str(e)}")
            raise ModelError(f"Frequency analysis failed: {str(e)}")

    def _find_dominant_frequencies(self, frequencies, power_spectrum, threshold=0.1):
        """Find dominant frequencies in the spectrum."""
        max_power = np.max(power_spectrum)
        dominant_indices = np.where(power_spectrum > threshold * max_power)[0]
        return frequencies[dominant_indices].tolist()

    def _analyze_harmonics(self, frequencies):
        """Analyze harmonic relationships between frequencies."""
        harmonics = {}
        for name, sacred_freqs in self.sacred_frequencies.items():
            harmonics[name] = []
            for freq in frequencies:
                for sacred_freq in sacred_freqs:
                    ratio = freq / sacred_freq
                    if 0.95 <= ratio <= 1.05:  # 5% tolerance
                        harmonics[name].append({
                            "frequency": freq,
                            "sacred_frequency": sacred_freq,
                            "ratio": ratio
                        })
        return harmonics

    def _calculate_resonance(self, frequencies):
        """Calculate resonance patterns between frequencies."""
        resonance = []
        for i in range(len(frequencies)):
            for j in range(i+1, len(frequencies)):
                ratio = frequencies[j] / frequencies[i]
                for name, ratios in self.harmonic_ratios.items():
                    for sacred_ratio in ratios:
                        if 0.95 <= ratio/sacred_ratio <= 1.05:  # 5% tolerance
                            resonance.append({
                                "frequency1": frequencies[i],
                                "frequency2": frequencies[j],
                                "ratio": ratio,
                                "sacred_ratio": sacred_ratio,
                                "pattern": name
                            })
        return resonance

    def generate_resonant_signal(self, base_frequency, duration=1.0, sample_rate=44100):
        """Generate a resonant signal based on sacred frequencies."""
        try:
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Generate base signal
            signal = np.sin(2 * np.pi * base_frequency * t)
            
            # Add harmonics based on sacred frequencies
            for name, frequencies in self.sacred_frequencies.items():
                for freq in frequencies:
                    if freq != base_frequency:
                        harmonic = np.sin(2 * np.pi * freq * t)
                        signal += 0.1 * harmonic  # Add with reduced amplitude
            
            return {
                "time": t.tolist(),
                "signal": signal.tolist(),
                "base_frequency": base_frequency,
                "duration": duration,
                "sample_rate": sample_rate
            }
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            raise ModelError(f"Signal generation failed: {str(e)}")

    def visualize_spectrum(self, frequencies, power_spectrum, save_path=None):
        """Visualize the frequency spectrum."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(frequencies, power_spectrum)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.title('Frequency Spectrum')
            plt.grid(True)
            
            # Mark sacred frequencies
            for name, sacred_freqs in self.sacred_frequencies.items():
                for freq in sacred_freqs:
                    plt.axvline(x=freq, color='r', linestyle='--', alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Spectrum visualization failed: {str(e)}")
            raise ModelError(f"Spectrum visualization failed: {str(e)}") 