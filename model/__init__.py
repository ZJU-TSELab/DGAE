import importlib
from pathlib import Path
from ._model_register import ModelRegistry


# Automatically import all models from the current directory
def import_models():
    """
    Automatically discover and import all model files in the model directory.
    A file is considered a model file if it:
    1. Ends with .py
    2. Is not a special Python file (doesn't start with _)
    3. Is not this file (__init__.py)
    """
    current_dir = Path(__file__).parent
    for file in current_dir.glob("*.py"):
        if file.name.startswith("_"):  # Skip __init__.py and other special files
            continue

        module_name = file.stem  # Get filename without extension
        importlib.import_module(f".{module_name}", package="model")


# Import all models when this module is imported
import_models()


# Expose the registry and convenience functions
def get_model(name):
    """Get a model by name"""
    return ModelRegistry.get_model(name)


def list_models():
    """List all available models"""
    return ModelRegistry.list_models()


# Export main interfaces
__all__ = ['ModelRegistry', 'get_model', 'list_models']
