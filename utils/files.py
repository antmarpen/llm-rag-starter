import inspect
import pkgutil
import importlib
from typing import Type

def get_all_subclasses(base_class: Type, package_name: str):
    subclasses = []
    # Iterate over all modules in the package
    for _, module_name, is_pkg in pkgutil.walk_packages(path=importlib.import_module(package_name).__path__, prefix=f"{package_name}."):
        if not is_pkg:
            module = importlib.import_module(module_name)
            # Loop the module members
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if the classes is a subclass of the base class
                if issubclass(obj, base_class) and obj is not base_class:
                    subclasses.append(obj)
    return subclasses