import builtins
import importlib
import sys
from contextlib import contextmanager

import pytest


@pytest.fixture
def fresh_import():
    imported = []

    def _fresh_import(module_name: str):
        imported.append(module_name)
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    yield _fresh_import

    for module_name in imported:
        sys.modules.pop(module_name, None)


@pytest.fixture
def blocked_import():
    @contextmanager
    def _blocked_import(module_name: str):
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == module_name or name.startswith(f"{module_name}."):
                raise ImportError(f"blocked import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        try:
            yield
        finally:
            builtins.__import__ = original_import

    return _blocked_import