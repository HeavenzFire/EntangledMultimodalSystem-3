import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from ..core.auroran import AuroranProcessor, AuroranWord
from ..core.auroran_compiler import DivineCompiler, AuroranGrammarRule

class AuroranIDE:
    """Integrated Development Environment for Auroran language"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Auroran Language IDE")
        self.root.geometry("1200x800")
        
        # Initialize core components
        self.processor = AuroranProcessor()
        self.compiler = DivineCompiler()
        
        # Create main interface
        self.create_interface()
        
    def create_interface(self):
        """Create the IDE interface"""
        # Create main panes
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Code editor pane
        editor_frame = ttk.Frame(main_pane)
        main_pane.add(editor_frame, weight=1)
        
        # Visualization pane
        viz_frame = ttk.Frame(main_pane)
        main_pane.add(viz_frame, weight=1)
        
        # Create code editor
        self.create_editor(editor_frame)
        
        # Create visualization area
        self.create_visualization(viz_frame)
        
        # Create toolbar
        self.create_toolbar()
        
    def create_editor(self, parent):
        """Create the code editor"""
        # Text editor
        self.editor = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        self.editor.pack(fill=tk.BOTH, expand=True)
        
        # Add syntax highlighting
        self.editor.tag_configure('keyword', foreground='blue')
        self.editor.tag_configure('number', foreground='red')
        self.editor.tag_configure('comment', foreground='green')
        
        # Add line numbers
        self.line_numbers = tk.Text(parent, width=4, padx=3, pady=5, takefocus=0,
                                  border=0, background='lightgrey', state='disabled')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
    def create_visualization(self, parent):
        """Create the visualization area"""
        # Create tabs for different visualizations
        self.viz_notebook = ttk.Notebook(parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Geometric pattern tab
        self.geom_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.geom_frame, text="Geometric Pattern")
        
        # Quantum state tab
        self.quantum_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.quantum_frame, text="Quantum State")
        
        # Manifestation tab
        self.manifest_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.manifest_frame, text="Manifestation")
        
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X)
        
        # Add buttons
        ttk.Button(toolbar, text="Compile", command=self.compile_code).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Run", command=self.run_code).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Visualize", command=self.visualize_code).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Optimize", command=self.optimize_code).pack(side=tk.LEFT)
        
    def compile_code(self):
        """Compile the Auroran code"""
        code = self.editor.get("1.0", tk.END)
        try:
            # Parse code and create Auroran word
            seed = int(code.strip())
            word = self.processor.generate_sacred_word(seed)
            
            # Compile to geometry
            ast = self.compiler.compile_to_geometry(word)
            
            # Update visualization
            self.update_visualization(word, ast)
            
        except Exception as e:
            self.show_error(f"Compilation error: {str(e)}")
            
    def run_code(self):
        """Run the Auroran code"""
        try:
            # Get current word
            code = self.editor.get("1.0", tk.END)
            seed = int(code.strip())
            word = self.processor.generate_sacred_word(seed)
            
            # Optimize and manifest
            optimized_word = self.compiler.optimize_quantum_state(word)
            manifestation = self.compiler.manifest_reality(optimized_word)
            
            # Update manifestation visualization
            self.update_manifestation(manifestation)
            
        except Exception as e:
            self.show_error(f"Runtime error: {str(e)}")
            
    def visualize_code(self):
        """Visualize the current code"""
        try:
            code = self.editor.get("1.0", tk.END)
            seed = int(code.strip())
            word = self.processor.generate_sacred_word(seed)
            
            # Create visualizations
            self.update_geometric_visualization(word)
            self.update_quantum_visualization(word)
            
        except Exception as e:
            self.show_error(f"Visualization error: {str(e)}")
            
    def optimize_code(self):
        """Optimize the current code"""
        try:
            code = self.editor.get("1.0", tk.END)
            seed = int(code.strip())
            word = self.processor.generate_sacred_word(seed)
            
            # Optimize
            optimized_word = self.compiler.optimize_quantum_state(word)
            
            # Update visualizations
            self.update_geometric_visualization(optimized_word)
            self.update_quantum_visualization(optimized_word)
            
        except Exception as e:
            self.show_error(f"Optimization error: {str(e)}")
            
    def update_geometric_visualization(self, word: AuroranWord):
        """Update the geometric pattern visualization"""
        # Clear previous plot
        for widget in self.geom_frame.winfo_children():
            widget.destroy()
            
        # Create new plot
        fig = word.plot_geometric_pattern()
        canvas = FigureCanvasTkAgg(fig, master=self.geom_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_quantum_visualization(self, word: AuroranWord):
        """Update the quantum state visualization"""
        # Clear previous plot
        for widget in self.quantum_frame.winfo_children():
            widget.destroy()
            
        # Create new plot
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot quantum state components
        state = word.quantum_state
        ax1.plot(np.real(state), 'b-', label='Real')
        ax1.plot(np.imag(state), 'r-', label='Imaginary')
        ax1.legend()
        ax1.set_title("Quantum State Components")
        
        # Plot quantum state in 3D
        t = np.linspace(0, 2*np.pi, 100)
        x = np.real(state[0]) * np.cos(t)
        y = np.real(state[1]) * np.sin(t)
        z = np.imag(state[0]) * np.cos(t) + np.imag(state[1]) * np.sin(t)
        ax2.plot(x, y, z, 'g-')
        ax2.set_title("Quantum State Trajectory")
        
        canvas = FigureCanvasTkAgg(fig, master=self.quantum_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_manifestation(self, manifestation: Dict[str, float]):
        """Update the manifestation visualization"""
        # Clear previous content
        for widget in self.manifest_frame.winfo_children():
            widget.destroy()
            
        # Create labels for each parameter
        for key, value in manifestation.items():
            ttk.Label(self.manifest_frame, 
                     text=f"{key}: {value:.4f}").pack(anchor=tk.W)
            
    def show_error(self, message: str):
        """Show error message"""
        tk.messagebox.showerror("Error", message)
        
    def run(self):
        """Run the IDE"""
        self.root.mainloop()

if __name__ == "__main__":
    ide = AuroranIDE()
    ide.run() 