"""
Universal Technology Registry — supports dynamic discovery, loading, and orchestration
of any module (AI, cloud, neural, quantum, IoT, blockchain, web, social, etc).
"""

from typing import Dict, Any, List, Optional, Callable, Union
import logging
import threading
import importlib
import inspect
from pathlib import Path
import json
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalRegistry:
    """Universal registry for all technology modules"""
    
    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load registry configuration"""
        config_path = Path("config/registry_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "auto_discover": True,
                "dependency_check": True,
                "version_check": True
            }
    
    def register(self, 
                name: str, 
                module: Any, 
                metadata: Optional[Dict[str, Any]] = None,
                dependencies: Optional[List[str]] = None) -> None:
        """
        Register a new technology module
        
        Args:
            name: Unique identifier for the module
            module: The module to register
            metadata: Optional metadata about the module
            dependencies: Optional list of required dependencies
        """
        with self.lock:
            if name in self.modules:
                logger.warning(f"Module {name} already registered, overwriting")
            
            self.modules[name] = module
            self.metadata[name] = metadata or {}
            self.dependencies[name] = dependencies or []
            
            logger.info(f"Registered module: {name}")
            if metadata:
                logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
            if dependencies:
                logger.info(f"Dependencies: {dependencies}")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a registered module by name
        
        Args:
            name: Name of the module to retrieve
            
        Returns:
            Optional[Any]: The module if found, None otherwise
        """
        with self.lock:
            return self.modules.get(name)
    
    def list_all(self) -> List[str]:
        """
        List all registered module names
        
        Returns:
            List[str]: List of all registered module names
        """
        with self.lock:
            return list(self.modules.keys())
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a module
        
        Args:
            name: Name of the module
            
        Returns:
            Optional[Dict[str, Any]]: Module metadata if found, None otherwise
        """
        with self.lock:
            return self.metadata.get(name)
    
    def check_dependencies(self, name: str) -> bool:
        """
        Check if all dependencies for a module are satisfied
        
        Args:
            name: Name of the module
            
        Returns:
            bool: True if all dependencies are satisfied, False otherwise
        """
        with self.lock:
            if name not in self.dependencies:
                return True
            
            for dep in self.dependencies[name]:
                if dep not in self.modules:
                    logger.warning(f"Dependency {dep} not found for module {name}")
                    return False
            return True
    
    def compose(self, 
                sequence: List[Union[str, tuple]], 
                context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Compose and execute a workflow sequence
        
        Args:
            sequence: List of module names or (name, params) tuples
            context: Optional context to pass between modules
            
        Returns:
            Any: Final output of the workflow
        """
        output = context or {}
        
        for step in sequence:
            if isinstance(step, str):
                name, params = step, {}
            else:
                name, params = step
            
            module = self.get(name)
            if not module:
                logger.error(f"Module {name} not found")
                return {"error": f"Module {name} not found"}
            
            # Check dependencies
            if not self.check_dependencies(name):
                logger.error(f"Dependencies not satisfied for {name}")
                return {"error": f"Dependencies not satisfied for {name}"}
            
            # Prepare parameters
            if output:
                params = params or {}
                params["input"] = output
            
            try:
                # Execute module
                if callable(module):
                    output = module(**params)
                else:
                    output = module
                
                logger.info(f"Executed {name} successfully")
                
            except Exception as e:
                logger.error(f"Error executing {name}: {str(e)}")
                return {"error": f"Error executing {name}: {str(e)}"}
        
        return output
    
    def auto_discover(self, directory: str = "integrations") -> None:
        """
        Automatically discover and register modules from a directory
        
        Args:
            directory: Directory to search for modules
        """
        if not self.config.get("auto_discover"):
            return
        
        for path in Path(directory).rglob("*.py"):
            try:
                # Convert path to module name
                module_name = str(path).replace("/", ".").replace(".py", "")
                if module_name.endswith(".__init__"):
                    module_name = module_name[:-9]
                
                # Import module
                module = importlib.import_module(module_name)
                
                # Find registerable objects
                for name, obj in inspect.getmembers(module):
                    if (inspect.isfunction(obj) or inspect.isclass(obj)) and not name.startswith("_"):
                        # Register with metadata
                        metadata = {
                            "source": module_name,
                            "type": "function" if inspect.isfunction(obj) else "class"
                        }
                        self.register(f"{module_name}.{name}", obj, metadata)
                
            except Exception as e:
                logger.error(f"Error discovering module {path}: {str(e)}")

# Singleton instance for global use
_universal_registry = UniversalRegistry()

def get_registry() -> UniversalRegistry:
    """Get the global universal registry instance"""
    return _universal_registry

# Example usage
if __name__ == "__main__":
    # Create registry
    registry = get_registry()
    
    # Register some test modules
    def dummy_ai(**kwargs):
        return {"ai": "done", "args": kwargs}
    
    def dummy_cloud(**kwargs):
        return {"cloud": "done", "args": kwargs}
    
    registry.register(
        "dummy-ai",
        dummy_ai,
        metadata={"version": "1.0", "type": "ai"},
        dependencies=["dummy-cloud"]
    )
    
    registry.register(
        "dummy-cloud",
        dummy_cloud,
        metadata={"version": "1.0", "type": "cloud"}
    )
    
    # List modules
    print("Registered modules:", registry.list_all())
    
    # Get metadata
    print("AI metadata:", registry.get_metadata("dummy-ai"))
    
    # Check dependencies
    print("AI dependencies satisfied:", registry.check_dependencies("dummy-ai"))
    
    # Compose workflow
    sequence = [
        ("dummy-cloud", {"param": "value"}),
        ("dummy-ai", {"input": None})
    ]
    result = registry.compose(sequence)
    print("Workflow result:", result)
    
    # Auto-discover modules
    registry.auto_discover()
    print("After auto-discovery:", registry.list_all()) 