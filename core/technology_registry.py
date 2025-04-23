"""
Central Technology Registry for Infinite Integration.
Allows dynamic registration and discovery of any computational, sensor, or reasoning module.
"""

from typing import Dict, Any, Optional, List
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnologyRegistry:
    """Central registry for all technology modules"""
    
    def __init__(self):
        self.modules: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def register(self, name: str, module: Any) -> None:
        """
        Register a new technology module
        
        Args:
            name: Unique identifier for the module
            module: The module to register
        """
        with self.lock:
            if name in self.modules:
                logger.warning(f"Module {name} already registered, overwriting")
            self.modules[name] = module
            logger.info(f"Registered module: {name}")
    
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
    
    def remove(self, name: str) -> bool:
        """
        Remove a module from the registry
        
        Args:
            name: Name of the module to remove
            
        Returns:
            bool: True if module was removed, False if not found
        """
        with self.lock:
            if name in self.modules:
                del self.modules[name]
                logger.info(f"Removed module: {name}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all registered modules"""
        with self.lock:
            self.modules.clear()
            logger.info("Cleared all modules")

# Singleton instance for global use
_registry = TechnologyRegistry()

def get_registry() -> TechnologyRegistry:
    """Get the global technology registry instance"""
    return _registry

# Example usage
if __name__ == "__main__":
    class DummyModule:
        def __init__(self, name: str):
            self.name = name
        
        def __call__(self, *args, **kwargs):
            return f"Dummy module {self.name} called with {args}, {kwargs}"
    
    # Register some dummy modules
    registry = get_registry()
    registry.register("dummy1", DummyModule("module1"))
    registry.register("dummy2", DummyModule("module2"))
    
    # List all modules
    print("Registered modules:", registry.list_all())
    
    # Get and use a module
    module = registry.get("dummy1")
    if module:
        print(module("test", param="value"))
    
    # Remove a module
    registry.remove("dummy2")
    print("Remaining modules:", registry.list_all())
    
    # Clear all modules
    registry.clear()
    print("After clear:", registry.list_all()) 