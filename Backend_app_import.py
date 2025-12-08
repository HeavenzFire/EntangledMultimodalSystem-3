"""
Backend App Import Helper
This module provides access to the Flask application and background services from Backend app.py
"""

# Import from Backend app.py (handling the space in filename)
import importlib.util
import sys
import os

# Dynamically import the module with a space in its filename
module_name = "Backend_app"
file_path = os.path.join(os.path.dirname(__file__), "Backend app.py")

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access the exported symbols
app = module.app
run_app = module.run_app
mitigate_radiation = module.mitigate_radiation

# Export symbols for use in main orchestrator
__all__ = ['app', 'run_app', 'mitigate_radiation']