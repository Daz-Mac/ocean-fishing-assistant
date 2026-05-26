"""Test configuration — mocks homeassistant before package imports."""
import sys
from pathlib import Path

# Mock homeassistant modules before any package imports trigger them
import types

ha = types.ModuleType("homeassistant")
ha.const = types.ModuleType("homeassistant.const")
ha.const.CONF_LATITUDE = "latitude"
ha.const.CONF_LONGITUDE = "longitude"
ha.helpers = types.ModuleType("homeassistant.helpers")
ha.helpers.update_coordinator = types.ModuleType("homeassistant.helpers.update_coordinator")
ha.helpers.aiohttp_client = types.ModuleType("homeassistant.helpers.aiohttp_client")
ha.helpers.storage = types.ModuleType("homeassistant.helpers.storage")
ha.helpers.event = types.ModuleType("homeassistant.helpers.event")
ha.util = types.ModuleType("homeassistant.util")
ha.util.dt = types.ModuleType("homeassistant.util.dt")
ha.data_entry_flow = types.ModuleType("homeassistant.data_entry_flow")

sys.modules["homeassistant"] = ha
sys.modules["homeassistant.const"] = ha.const
sys.modules["homeassistant.helpers"] = ha.helpers
sys.modules["homeassistant.helpers.update_coordinator"] = ha.helpers.update_coordinator
sys.modules["homeassistant.helpers.aiohttp_client"] = ha.helpers.aiohttp_client
sys.modules["homeassistant.helpers.storage"] = ha.helpers.storage
sys.modules["homeassistant.helpers.event"] = ha.helpers.event
sys.modules["homeassistant.util"] = ha.util
sys.modules["homeassistant.util.dt"] = ha.util.dt
sys.modules["homeassistant.data_entry_flow"] = ha.data_entry_flow

# Mock external dependencies
for mod_name in ("aiohttp", "skyfield", "voluptuous", "zoneinfo"):
    sys.modules[mod_name] = types.ModuleType(mod_name)

# Add custom_components to path
CUSTOM_COMPONENTS = Path(__file__).resolve().parent.parent / "custom_components"
if str(CUSTOM_COMPONENTS) not in sys.path:
    sys.path.insert(0, str(CUSTOM_COMPONENTS))
