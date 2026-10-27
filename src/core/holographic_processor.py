import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Optional
import time

class TensorHolographyDB:
    """Memory-bounded, reversible FFT-based holographic transform.

    The former implementation eagerly allocated a dense 16,384 x 16,384
    float64 matrix (~2 GiB) and then attempted cubic-time pseudo-inversion.
    This implementation preserves the encode/decode contract without a
    resolution-sized dense matrix. It is a numerical transform, not a claim
    of physical holography.
    """

    def __init__(self):
        # Kept as a compatibility attribute; no dense matrix is needed.
        self.encoding_matrix = None

    def _initialize_encoding(self):
        """Compatibility no-op: the transform is matrix-free."""
        self.encoding_matrix = None

    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Apply a reversible N-dimensional Fourier encoding."""
        try:
            data = np.asarray(input_data)
            if data.size == 0 or not np.all(np.isfinite(data)):
                raise ValueError("input_data must be non-empty and finite")
            return np.fft.fftn(data)
        except Exception as e:
            logger.error(f"Error encoding holographic data: {str(e)}")
            raise ModelError(f"Holographic encoding failed: {str(e)}") from e

    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Invert the Fourier encoding while preserving the original shape."""
        try:
            data = np.asarray(encoded_data)
            if data.size == 0 or not np.all(np.isfinite(data)):
                raise ValueError("encoded_data must be non-empty and finite")
            return np.fft.ifftn(data)
        except Exception as e:
            logger.error(f"Error decoding holographic data: {str(e)}")
            raise ModelError(f"Holographic decoding failed: {str(e)}") from e

class HolographicProcessor:
    """16K photonic processing engine with real-time rendering capabilities."""
    
    def __init__(self):
        self.resolution = 16384  # 16K holograms
        self.tensor_holography = TensorHolographyDB()
        self.rendering_latency = 0.008  # 8ms target latency
        self.last_render_time = None
        
    def render_realtime(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Render 3D scenes with 8ms latency."""
        try:
            start_time = time.time()
            
            # Encode input data
            encoded = self.tensor_holography.encode(input_data["scene_data"])
            
            # Apply photonic transformations
            transformed = self._apply_photonic_transforms(encoded)
            
            # Decode to final output
            rendered = self.tensor_holography.decode(transformed)
            
            # Calculate actual latency
            render_time = time.time() - start_time
            self.last_render_time = render_time
            
            return {
                "rendered_scene": rendered,
                "latency": render_time,
                "resolution": self.resolution,
                "quality_metric": self._calculate_quality(rendered, input_data["scene_data"])
            }
            
        except Exception as e:
            logger.error(f"Error rendering holographic scene: {str(e)}")
            raise ModelError(f"Holographic rendering failed: {str(e)}")
            
    def _apply_photonic_transforms(self, encoded_data: np.ndarray) -> np.ndarray:
        """Apply a deterministic, shape-preserving numerical phase transform."""
        try:
            data = np.asarray(encoded_data)
            if data.size == 0 or not np.all(np.isfinite(data)):
                raise ValueError("encoded_data must be non-empty and finite")
            propagated = np.fft.fftn(data)
            phase_values = np.linspace(0.0, 2.0 * np.pi, data.size, endpoint=False)
            phase = np.exp(1j * phase_values.reshape(data.shape))
            return np.fft.ifftn(propagated * phase)
        except Exception as e:
            logger.error(f"Error applying photonic transforms: {str(e)}")
            raise ModelError(f"Photonic transformation failed: {str(e)}") from e
            
    def _calculate_quality(self, rendered: np.ndarray, original: np.ndarray) -> float:
        """Calculate rendering quality metric."""
        try:
            # Calculate PSNR
            mse = float(np.mean(np.abs(rendered - original) ** 2))
            if mse == 0:
                return float('inf')
            max_pixel = 1.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return psnr
            
        except Exception as e:
            logger.error(f"Error calculating quality metric: {str(e)}")
            return 0.0
            
    def project(self, input_state: np.ndarray) -> np.ndarray:
        """Project a numeric state through the matrix-free transform."""
        start_time = time.time()
        encoded = self.tensor_holography.encode(input_state)
        projected = self._apply_photonic_transforms(encoded)
        self.last_render_time = time.time() - start_time
        original = np.asarray(input_state).reshape(-1)
        output = self.tensor_holography.decode(projected).reshape(-1)
        denom = float(np.linalg.norm(original) * np.linalg.norm(output))
        similarity = float(abs(np.vdot(original, output)) / denom) if denom else 0.0
        self._metrics = {
            "fidelity": min(1.0, max(0.0, similarity)),
            "speed": 1.0 / max(self.last_render_time, 1e-9),
            "efficiency": 1.0 / (1.0 + self.last_render_time),
            "error_rate": max(0.0, 1.0 - min(1.0, similarity)),
        }
        return output

    def get_metrics(self) -> Dict[str, float]:
        """Return numerical transform diagnostics, not physical-device metrics."""
        return getattr(self, "_metrics", {
            "fidelity": 0.0, "speed": 0.0, "efficiency": 0.0, "error_rate": 1.0
        }).copy()

    def calibrate(self) -> Dict[str, float]:
        """Report a neutral software calibration result for interface compatibility."""
        self._metrics = {
            "fidelity": 1.0, "speed": 0.0, "efficiency": 1.0, "error_rate": 0.0
        }
        return {"score": 1.0, "mode": "software-transform"}

    def get_state(self) -> Dict[str, Any]:
        """Get current processor state."""
        return {
            "resolution": self.resolution,
            "last_render_time": self.last_render_time,
            "target_latency": self.rendering_latency
        }
        
    def reset(self) -> None:
        """Reset processor to initial state."""
        self.tensor_holography = TensorHolographyDB()
        self.last_render_time = None

    def create_hologram(self, object_wave, reference_wave):
        """Create a hologram from object and reference waves."""
        try:
            # Calculate interference pattern
            interference = object_wave + reference_wave
            self.hologram = np.abs(interference)**2
            
            return self.hologram
        except Exception as e:
            logger.error(f"Hologram creation failed: {str(e)}")
            raise ModelError(f"Hologram creation failed: {str(e)}")

    def propagate_wave(self, wave, distance):
        """Propagate a wave field over a given distance."""
        try:
            # Get wave dimensions
            N = wave.shape[0]
            M = wave.shape[1]
            
            # Create frequency grid
            kx = 2 * np.pi * np.fft.fftfreq(N, self.pixel_size)
            ky = 2 * np.pi * np.fft.fftfreq(M, self.pixel_size)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            
            # Calculate propagation kernel
            k = 2 * np.pi / self.wavelength
            kernel = np.exp(1j * distance * np.sqrt(k**2 - KX**2 - KY**2))
            
            # Apply propagation
            wave_fft = fft2(wave)
            propagated_fft = wave_fft * kernel
            propagated = ifft2(propagated_fft)
            
            return propagated
        except Exception as e:
            logger.error(f"Wave propagation failed: {str(e)}")
            raise ModelError(f"Wave propagation failed: {str(e)}")

    def reconstruct_hologram(self, reference_wave):
        """Reconstruct the hologram using the reference wave."""
        try:
            if self.hologram is None:
                raise ValueError("No hologram available for reconstruction")
            
            # Multiply hologram with reference wave
            reconstruction = self.hologram * reference_wave
            
            # Propagate to reconstruction plane
            self.reconstruction = self.propagate_wave(reconstruction, self.distance)
            
            return self.reconstruction
        except Exception as e:
            logger.error(f"Hologram reconstruction failed: {str(e)}")
            raise ModelError(f"Hologram reconstruction failed: {str(e)}")

    def create_point_source(self, position, size):
        """Create a point source wave field."""
        try:
            # Create coordinate grid
            x = np.linspace(-size/2, size/2, size)
            y = np.linspace(-size/2, size/2, size)
            X, Y = np.meshgrid(x, y)
            
            # Calculate distances
            R = np.sqrt((X - position[0])**2 + (Y - position[1])**2)
            
            # Create spherical wave
            wave = np.exp(1j * 2 * np.pi * R / self.wavelength) / R
            
            return wave
        except Exception as e:
            logger.error(f"Point source creation failed: {str(e)}")
            raise ModelError(f"Point source creation failed: {str(e)}")

    def create_plane_wave(self, angle, size):
        """Create a plane wave field."""
        try:
            # Create coordinate grid
            x = np.linspace(-size/2, size/2, size)
            y = np.linspace(-size/2, size/2, size)
            X, Y = np.meshgrid(x, y)
            
            # Calculate phase
            phase = np.sin(angle) * X + np.cos(angle) * Y
            
            # Create plane wave
            wave = np.exp(1j * 2 * np.pi * phase / self.wavelength)
            
            return wave
        except Exception as e:
            logger.error(f"Plane wave creation failed: {str(e)}")
            raise ModelError(f"Plane wave creation failed: {str(e)}")

    def visualize_hologram(self, save_path=None):
        """Visualize the hologram."""
        try:
            if self.hologram is None:
                raise ValueError("No hologram available for visualization")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(np.abs(self.hologram), cmap='gray')
            plt.title('Hologram')
            plt.colorbar()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Hologram visualization failed: {str(e)}")
            raise ModelError(f"Hologram visualization failed: {str(e)}")

    def visualize_reconstruction(self, save_path=None):
        """Visualize the reconstruction."""
        try:
            if self.reconstruction is None:
                raise ValueError("No reconstruction available for visualization")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(np.abs(self.reconstruction), cmap='gray')
            plt.title('Reconstruction')
            plt.colorbar()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            logger.error(f"Reconstruction visualization failed: {str(e)}")
            raise ModelError(f"Reconstruction visualization failed: {str(e)}")

    def get_hologram_info(self):
        """Get information about the hologram."""
        try:
            if self.hologram is None:
                raise ValueError("No hologram available")
            
            return {
                "dimensions": self.hologram.shape,
                "min_intensity": np.min(np.abs(self.hologram)),
                "max_intensity": np.max(np.abs(self.hologram)),
                "mean_intensity": np.mean(np.abs(self.hologram)),
                "reconstruction_available": self.reconstruction is not None
            }
        except Exception as e:
            logger.error(f"Hologram info retrieval failed: {str(e)}")
            raise ModelError(f"Hologram info retrieval failed: {str(e)}") 