import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Optional
import time

class TensorHolographyDB:
    """Tensor-based holographic encoding and decoding."""
    
    def __init__(self):
        self.encoding_matrix = None
        self._initialize_encoding()
        
    def _initialize_encoding(self):
        """Initialize holographic encoding matrix."""
        try:
            # Create encoding matrix for 16K resolution
            self.encoding_matrix = np.random.randn(16384, 16384)
            # Normalize matrix
            self.encoding_matrix /= np.linalg.norm(self.encoding_matrix)
            
        except Exception as e:
            logger.error(f"Error initializing encoding matrix: {str(e)}")
            raise ModelError(f"Encoding initialization failed: {str(e)}")
            
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Encode input data into holographic format."""
        try:
            # Ensure input matches encoding dimensions
            if input_data.shape != (16384, 16384):
                input_data = np.resize(input_data, (16384, 16384))
                
            # Apply encoding transformation
            encoded = np.dot(self.encoding_matrix, input_data)
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding holographic data: {str(e)}")
            raise ModelError(f"Holographic encoding failed: {str(e)}")
            
    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """Decode holographic data back to original format."""
        try:
            # Apply inverse transformation
            decoded = np.dot(np.linalg.pinv(self.encoding_matrix), encoded_data)
            return decoded
            
        except Exception as e:
            logger.error(f"Error decoding holographic data: {str(e)}")
            raise ModelError(f"Holographic decoding failed: {str(e)}")

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
        """Apply photonic transformations to encoded data."""
        try:
            # Apply wavefront propagation
            propagated = np.fft.fft2(encoded_data)
            
            # Apply phase modulation
            phase = np.exp(1j * np.random.rand(*encoded_data.shape))
            modulated = propagated * phase
            
            # Inverse transform
            transformed = np.fft.ifft2(modulated)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error applying photonic transforms: {str(e)}")
            raise ModelError(f"Photonic transformation failed: {str(e)}")
            
    def _calculate_quality(self, rendered: np.ndarray, original: np.ndarray) -> float:
        """Calculate rendering quality metric."""
        try:
            # Calculate PSNR
            mse = np.mean((rendered - original) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 1.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return psnr
            
        except Exception as e:
            logger.error(f"Error calculating quality metric: {str(e)}")
            return 0.0
            
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