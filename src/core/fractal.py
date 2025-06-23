import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import logger
from src.utils.errors import ModelError
from src.config import Config

class FractalGenerator:
    def __init__(self):
        """Initialize the fractal generator."""
        try:
            logger.info("FractalGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FractalGenerator: {str(e)}")
            raise ModelError(f"Fractal generator initialization failed: {str(e)}")

    def generate_mandelbrot(self, width=800, height=800, max_iter=100):
        """Generate a Mandelbrot set fractal."""
        try:
            x = np.linspace(-2, 1, width)
            y = np.linspace(-1.5, 1.5, height)
            X, Y = np.meshgrid(x, y)
            c = X + 1j * Y
            z = np.zeros_like(c)
            divtime = np.zeros(z.shape, dtype=int)
            
            for i in range(max_iter):
                z = z**2 + c
                diverge = z * np.conj(z) > 2**2
                div_now = diverge & (divtime == 0)
                divtime[div_now] = i
                z[diverge] = 2
            
            plt.figure(figsize=(10, 10))
            plt.imshow(divtime, cmap='inferno')
            plt.axis('off')
            plt.savefig('static/mandelbrot.png')
            plt.close()
            
            logger.info("Mandelbrot fractal generated successfully")
            return 'static/mandelbrot.png'
        except Exception as e:
            logger.error(f"Error in generate_mandelbrot method: {str(e)}")
            raise ModelError(f"Mandelbrot generation failed: {str(e)}")

    def generate_julia(self, c=-0.7 + 0.27j, width=800, height=800, max_iter=100):
        """Generate a Julia set fractal."""
        try:
            x = np.linspace(-2, 2, width)
            y = np.linspace(-2, 2, height)
            X, Y = np.meshgrid(x, y)
            z = X + 1j * Y
            divtime = np.zeros(z.shape, dtype=int)
            
            for i in range(max_iter):
                z = z**2 + c
                diverge = z * np.conj(z) > 2**2
                div_now = diverge & (divtime == 0)
                divtime[div_now] = i
                z[diverge] = 2
            
            plt.figure(figsize=(10, 10))
            plt.imshow(divtime, cmap='viridis')
            plt.axis('off')
            plt.savefig('static/julia.png')
            plt.close()
            
            logger.info("Julia fractal generated successfully")
            return 'static/julia.png'
        except Exception as e:
            logger.error(f"Error in generate_julia method: {str(e)}")
            raise ModelError(f"Julia set generation failed: {str(e)}")

    def generate_ifs(self, iterations=100000):
        """Generate an Iterated Function System fractal."""
        try:
            # Barnsley fern parameters
            functions = [
                (0.85, 0.04, -0.04, 0.85, 0, 1.6, 0.85),
                (0.2, -0.26, 0.23, 0.22, 0, 1.6, 0.07),
                (-0.15, 0.28, 0.26, 0.24, 0, 0.44, 0.07),
                (0, 0, 0, 0.16, 0, 0, 0.01)
            ]
            
            x, y = 0, 0
            points = []
            
            for _ in range(iterations):
                r = np.random.random()
                a, b, c, d, e, f, p = functions[0]
                if r > p:
                    for func in functions[1:]:
                        if r > p:
                            a, b, c, d, e, f, p = func
                        else:
                            break
                
                x, y = a * x + b * y + e, c * x + d * y + f
                points.append((x, y))
            
            points = np.array(points)
            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 0], points[:, 1], s=0.1, c='green')
            plt.axis('off')
            plt.savefig('static/ifs.png')
            plt.close()
            
            logger.info("IFS fractal generated successfully")
            return 'static/ifs.png'
        except Exception as e:
            logger.error(f"Error in generate_ifs method: {str(e)}")
            raise ModelError(f"IFS generation failed: {str(e)}")

# Create a global instance
fractal_generator = FractalGenerator()

def generate_fractal():
    """Global function to generate a fractal using the fractal generator."""
    return fractal_generator.generate_mandelbrot() 