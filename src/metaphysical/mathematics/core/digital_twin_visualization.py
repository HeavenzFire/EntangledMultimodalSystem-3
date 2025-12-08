import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from .digital_twin import DigitalTwin, DigitalTwinState

class DigitalTwinVisualizer:
    """Visualizes digital twin data across all biological scales"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_life_cycle(self, twin: DigitalTwin) -> plt.Figure:
        """Plot the digital twin's life cycle progression"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Extract timeline data
        ages = [state.age for _, state in twin.timeline]
        vitality = [state.health_metrics['vitality'] for _, state in twin.timeline]
        disease_risk = [state.disease_risk['cancer'] for _, state in twin.timeline]
        
        # Plot metrics
        self.ax.plot(ages, vitality, label='Vitality', color='green')
        self.ax.plot(ages, disease_risk, label='Cancer Risk', color='red')
        
        self.ax.set_title('Life Cycle Progression')
        self.ax.set_xlabel('Age (years)')
        self.ax.set_ylabel('Metric Value')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_organ_function(self, twin: DigitalTwin) -> plt.Figure:
        """Plot organ function metrics"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Get current organ function
        organ_function = twin.get_organ_function()
        
        # Create bar plot
        organs = list(organ_function.keys())
        values = list(organ_function.values())
        
        self.ax.bar(organs, values, color='blue')
        
        self.ax.set_title('Organ Function Metrics')
        self.ax.set_xlabel('Organ')
        self.ax.set_ylabel('Function Level')
        self.ax.grid(True)
        
        return self.fig
        
    def plot_cellular_metrics(self, twin: DigitalTwin) -> plt.Figure:
        """Plot cellular-scale metrics"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Get cellular metrics
        metrics = twin.get_cellular_metrics()
        
        # Create line plot
        t = np.linspace(0, 2*np.pi, 100)
        cell_cycle = np.sin(t + metrics['cell_cycle'] * 2*np.pi)
        differentiation = np.cos(t + metrics['differentiation'] * 2*np.pi)
        
        self.ax.plot(t, cell_cycle, label='Cell Cycle', color='blue')
        self.ax.plot(t, differentiation, label='Differentiation', color='green')
        self.ax.axhline(y=metrics['apoptosis'], color='red', linestyle='--', label='Apoptosis')
        
        self.ax.set_title('Cellular Metrics')
        self.ax.set_xlabel('Phase')
        self.ax.set_ylabel('Value')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_molecular_activity(self, twin: DigitalTwin) -> plt.Figure:
        """Plot molecular-scale activity"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Get molecular metrics
        metrics = twin.get_molecular_metrics()
        
        # Plot gene expression distribution
        self.ax.hist(metrics['gene_expression'], bins=50, alpha=0.5, label='Gene Expression')
        
        # Plot protein synthesis distribution
        self.ax.hist(metrics['protein_synthesis'], bins=50, alpha=0.5, label='Protein Synthesis')
        
        self.ax.set_title('Molecular Activity Distribution')
        self.ax.set_xlabel('Activity Level')
        self.ax.set_ylabel('Frequency')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_disease_progression(self, twin: DigitalTwin, disease_type: str) -> plt.Figure:
        """Plot disease progression metrics"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Extract disease-specific data from timeline
        ages = [state.age for _, state in twin.timeline]
        risk = [state.disease_risk[disease_type] for _, state in twin.timeline]
        
        # Plot risk progression
        self.ax.plot(ages, risk, label=f'{disease_type} Risk', color='red')
        
        # Add organ function if relevant
        if disease_type == 'cardiovascular':
            heart_function = [state.organ_function['heart'] for _, state in twin.timeline]
            self.ax.plot(ages, heart_function, label='Heart Function', color='blue')
        elif disease_type == 'neurodegenerative':
            brain_function = [state.organ_function['brain'] for _, state in twin.timeline]
            self.ax.plot(ages, brain_function, label='Brain Function', color='blue')
            
        self.ax.set_title(f'{disease_type} Progression')
        self.ax.set_xlabel('Age (years)')
        self.ax.set_ylabel('Metric Value')
        self.ax.grid(True)
        self.ax.legend()
        
        return self.fig
        
    def plot_health_metrics(self, twin: DigitalTwin) -> plt.Figure:
        """Plot comprehensive health metrics"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Get health summary
        health = twin.get_health_summary()
        
        # Create radar plot
        metrics = ['Vitality', 'Resilience', 'Homeostasis']
        values = [health['vitality'], health['resilience'], health['homeostasis']]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        self.ax.plot(angles, values, label='Health Metrics')
        self.ax.fill(angles, values, alpha=0.25)
        
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(metrics)
        self.ax.set_title('Health Metrics Radar Plot')
        self.ax.grid(True)
        
        return self.fig 