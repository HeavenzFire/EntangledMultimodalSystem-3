import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .light_cosmogenesis import LightSingularity, LuminousSovereign, LightEconomy

class LightVisualizer:
    """Visualizes the harmonious light-based reality system"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_light_growth(self, singularity: LightSingularity, t: np.ndarray) -> plt.Figure:
        """Plot exponential light growth through harmony"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Calculate growth
        k = 3.69
        L0 = singularity.luminosity
        L = L0 * np.exp(k * t * singularity.harmony_coefficient)
        
        # Plot
        self.ax.plot(t, L, label='Harmonious Light Growth')
        self.ax.set_title('Exponential Growth Through Harmony')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Luminosity (lm/m²)')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_integration_level(self, level: float) -> plt.Figure:
        """Plot harmonious integration level"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Create pie chart
        labels = ['Harmonious Integration', 'Potential']
        sizes = [level*100, (1-level)*100]
        colors = ['gold', 'lightgoldenrodyellow']
        
        self.ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        self.ax.set_title('Harmonious Integration Level')
        
        return self.fig
        
    def plot_sovereign_metrics(self, sovereign: LuminousSovereign) -> plt.Figure:
        """Plot sovereign harmonious metrics"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Metrics
        metrics = {
            'Essence': 1.0,
            'Frequency': sovereign.frequency / 1000,  # Normalize
            'Integration': sovereign.integration_level,
            'Luminosity': sovereign.emit_light() / 1e6  # Normalize
        }
        
        # Plot
        self.ax.bar(metrics.keys(), metrics.values(), color='gold')
        self.ax.set_title('Sovereign Harmonious Metrics')
        self.ax.set_ylabel('Normalized Value')
        self.ax.grid(True)
        
        return self.fig
        
    def plot_economy_flow(self, economy: LightEconomy, addresses: List[str]) -> plt.Figure:
        """Plot harmonious photon economy flow"""
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Get balances and harmony
        balances = [economy.photon_balances.get(addr, 0) for addr in addresses]
        harmony = [economy.harmony_flow.get(addr, 1.0) for addr in addresses]
        
        # Plot
        width = 0.35
        x = np.arange(len(addresses))
        
        self.ax.bar(x - width/2, balances, width, label='Photon Balance', color='gold')
        self.ax.bar(x + width/2, harmony, width, label='Harmony Flow', color='lightgoldenrodyellow')
        
        self.ax.set_title('Harmonious Economy Flow')
        self.ax.set_xlabel('Address')
        self.ax.set_ylabel('Value')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(addresses, rotation=45)
        self.ax.legend()
        self.ax.grid(True)
        
        return self.fig
        
    def plot_validation_metrics(self, metrics: Dict[str, float]) -> plt.Figure:
        """Plot system validation through harmony"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Thresholds
        thresholds = {
            'Integration Level': 1.0,
            'Light Growth': 3.69,
            'Harmony Flow': 1.0
        }
        
        # Plot metrics and thresholds
        x = np.arange(len(metrics))
        width = 0.35
        
        self.ax.bar(x - width/2, metrics.values(), width, label='Current', color='gold')
        self.ax.bar(x + width/2, thresholds.values(), width, label='Threshold', color='lightgoldenrodyellow')
        
        self.ax.set_title('Harmonious Validation Metrics')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(metrics.keys())
        self.ax.legend()
        self.ax.grid(True)
        
        return self.fig 