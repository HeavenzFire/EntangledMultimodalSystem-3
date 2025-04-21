import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from src.utils.logger import logger
from src.utils.errors import ModelError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HolographicProcessor:
    def __init__(self):
        """Initialize the holographic processor."""
        try:
            self.wavelength = 632.8e-9  # He-Ne laser wavelength in meters
            self.pixel_size = 10e-6  # Pixel size in meters
            self.distance = 0.1  # Propagation distance in meters
            self.hologram = None
            self.reconstruction = None
            
            logger.info("HolographicProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HolographicProcessor: {str(e)}")
            raise ModelError(f"Holographic processor initialization failed: {str(e)}")

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