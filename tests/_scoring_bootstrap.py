"""Import bootstrap for loading ocean_scoring and data_formatter
without triggering the HA-dependent package __init__.py."""

import importlib.util
import sys
import types
from pathlib import Path

COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "custom_components"
PKG_DIR = COMPONENTS_DIR / "ocean_fishing_assistant"

_MODULES = ["const", "safety", "moon_utils", "unit_helpers", "ocean_scoring", "data_formatter"]


def _ensure_homeassistant_mocks():
    """Ensure HA module mocks exist (solo mode when conftest hasn't loaded)."""
    ha = sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
    ha.util = sys.modules.setdefault("homeassistant.util", types.ModuleType("homeassistant.util"))
    dt_mod = sys.modules.setdefault("homeassistant.util.dt", types.ModuleType("homeassistant.util.dt"))

    if not hasattr(dt_mod, "parse_datetime"):
        from datetime import datetime

        def _parse_datetime(dt_str):
            if dt_str is None:
                return None
            try:
                return datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
            except (ValueError, AttributeError, TypeError):
                return None
        dt_mod.parse_datetime = _parse_datetime


def _init_package():
    """Ensure ocean_fishing_assistant package exists in sys.modules."""
    name = "ocean_fishing_assistant"
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(PKG_DIR)]
    pkg.__file__ = str(PKG_DIR / "__init__.py")
    pkg.__package__ = name
    sys.modules[name] = pkg


def load_ocean_module(name: str):
    """Load a module from the ocean_fishing_assistant package via importlib."""
    if name not in _MODULES:
        raise ValueError(f"Unknown module: {name}; must be one of {_MODULES}")

    # data_formatter needs homeassistant.util.dt mocks
    if name == "data_formatter":
        _ensure_homeassistant_mocks()

    _init_package()

    full_name = f"ocean_fishing_assistant.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, PKG_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "ocean_fishing_assistant"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod
