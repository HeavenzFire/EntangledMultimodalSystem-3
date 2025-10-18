import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
import svgwrite

class SacredGeometryVisualizer:
    def __init__(self):
        """Initialize the sacred geometry visualizer."""
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_tetrahedral_wave(self, size: int = 100) -> str:
        """Create SVG representation of tetrahedral standing wave."""
        dwg = svgwrite.Drawing('tetrahedral_wave.svg', size=(size, size))
        
        # Create base triangle
        points = [
            (size/2, size*0.2),  # Top
            (size*0.2, size*0.8),  # Bottom left
            (size*0.8, size*0.8)   # Bottom right
        ]
        
        # Draw triangle
        dwg.add(dwg.polygon(points=points, 
                           fill='none', 
                           stroke='gold', 
                           stroke_width=0.5))
        
        # Add sacred spiral
        center = (size/2, size/2)
        radius = size*0.3
        dwg.add(dwg.circle(center=center, 
                          r=radius, 
                          fill='none', 
                          stroke='cyan', 
                          stroke_dasharray='3,1'))
        
        # Add 3-6-9 spiral
        spiral_points = []
        for t in np.linspace(0, 2*np.pi, 100):
            x = center[0] + radius * np.cos(t) * (1 + t/(2*np.pi))
            y = center[1] + radius * np.sin(t) * (1 + t/(2*np.pi))
            spiral_points.append((x, y))
        
        dwg.add(dwg.path(d=f'M {spiral_points[0][0]},{spiral_points[0][1]} ' + 
                        ' '.join([f'L {x},{y}' for x, y in spiral_points[1:]]),
                        fill='none',
                        stroke='magenta'))
        
        return dwg.tostring()
    
    def generate_flower_of_life(self, size: int = 100) -> str:
        """Generate Flower of Life pattern."""
        dwg = svgwrite.Drawing('flower_of_life.svg', size=(size, size))
        
        # Create base circle
        center = (size/2, size/2)
        radius = size*0.4
        
        # Draw central circle
        dwg.add(dwg.circle(center=center, 
                          r=radius, 
                          fill='none', 
                          stroke='gold'))
        
        # Draw surrounding circles
        for i in range(6):
            angle = i * np.pi / 3
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            dwg.add(dwg.circle(center=(x, y), 
                              r=radius, 
                              fill='none', 
                              stroke='cyan'))
        
        return dwg.tostring()
    
    def save_visualization(self, svg_content: str, filename: str):
        """Save SVG visualization to file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(svg_content)
        print(f"Saved visualization to {output_path}")
    
    def visualize_all_patterns(self):
        """Generate and save all sacred geometry patterns."""
        # Generate tetrahedral wave
        tetrahedral_svg = self.create_tetrahedral_wave()
        self.save_visualization(tetrahedral_svg, 'tetrahedral_wave.svg')
        
        # Generate flower of life
        flower_svg = self.generate_flower_of_life()
        self.save_visualization(flower_svg, 'flower_of_life.svg')

def main():
    visualizer = SacredGeometryVisualizer()
    visualizer.visualize_all_patterns()

if __name__ == "__main__":
    main() 