"""Species profiles loader and validator.

Provides:
- load_profiles() -> dict
- validate_profiles(profiles: dict) -> None (raises ValueError on invalid schema)
- get_species_options() -> list of dicts for SelectSelector options: [{'label': common_name, 'value': key}, ...]
"""
from typing import Dict, Any, List
import json
import pkgutil

def load_profiles() -> Dict[str, Any]:
    raw = pkgutil.get_data(__package__, "species_profiles.json")
    if not raw:
        raise ValueError("species_profiles.json not found in package")
    try:
        profiles = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse species_profiles.json: {exc}")
    return profiles

def _is_number_list(v: Any, length: int = 2) -> bool:
    if not isinstance(v, list) or len(v) != length:
        return False
    for x in v:
        if not isinstance(x, (int, float)):
            return False
    return True

def validate_profiles(profiles: Dict[str, Any]) -> None:
    """Validate the loaded profiles. Raise ValueError on any schema problem."""
    if not isinstance(profiles, dict):
        raise ValueError("species_profiles.json root must be an object mapping keys to profiles")

    for key, profile in profiles.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid species key: {key!r}")
        if not isinstance(profile, dict):
            raise ValueError(f"Profile for '{key}' must be an object")

        # common_name required
        cn = profile.get("common_name")
        if not isinstance(cn, str) or not cn:
            raise ValueError(f"Profile '{key}': missing or invalid 'common_name'")

        # preferred_temp_c: list of two numbers
        if "preferred_temp_c" in profile and not _is_number_list(profile["preferred_temp_c"], 2):
            raise ValueError(f"Profile '{key}': 'preferred_temp_c' must be a list of two numbers")

        # preferred_wind_m_s: list of two numbers
        if "preferred_wind_m_s" in profile and not _is_number_list(profile["preferred_wind_m_s"], 2):
            raise ValueError(f"Profile '{key}': 'preferred_wind_m_s' must be a list of two numbers")

        # max_wave_height_m: number
        if "max_wave_height_m" in profile and not isinstance(profile["max_wave_height_m"], (int, float)):
            raise ValueError(f"Profile '{key}': 'max_wave_height_m' must be a number")

        # preferred_tide_m: list of two numbers
        if "preferred_tide_m" in profile and not _is_number_list(profile["preferred_tide_m"], 2):
            raise ValueError(f"Profile '{key}': 'preferred_tide_m' must be a list of two numbers")

        # preferred_tide_phase: list
        if "preferred_tide_phase" in profile and not isinstance(profile["preferred_tide_phase"], list):
            raise ValueError(f"Profile '{key}': 'preferred_tide_phase' must be a list")

        # preferred_times: list of dicts with start_hour and end_hour
        if "preferred_times" in profile:
            pt = profile["preferred_times"]
            if not isinstance(pt, list):
                raise ValueError(f"Profile '{key}': 'preferred_times' must be a list")
            for idx, t in enumerate(pt):
                if not isinstance(t, dict):
                    raise ValueError(f"Profile '{key}': preferred_times[{idx}] must be an object")
                sh = t.get("start_hour")
                eh = t.get("end_hour")
                if not isinstance(sh, int) or not (0 <= sh <= 23):
                    raise ValueError(f"Profile '{key}': preferred_times[{idx}].start_hour must be int 0..23")
                if not isinstance(eh, int) or not (0 <= eh <= 23):
                    raise ValueError(f"Profile '{key}': preferred_times[{idx}].end_hour must be int 0..23")

        # preferred_months: list of ints 1..12
        if "preferred_months" in profile:
            pm = profile["preferred_months"]
            if not isinstance(pm, list):
                raise ValueError(f"Profile '{key}': 'preferred_months' must be a list")
            for m in pm:
                if not isinstance(m, int) or not (1 <= m <= 12):
                    raise ValueError(f"Profile '{key}': preferred_months contains invalid month: {m!r}")

        # moon_preference: list
        if "moon_preference" in profile and not isinstance(profile["moon_preference"], list):
            raise ValueError(f"Profile '{key}': 'moon_preference' must be a list")

def get_species_options(profiles: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return options suitable for Home Assistant SelectSelectorConfig.options

    Each option is a dict with 'label' and 'value'. Sorted by label.
    """
    opts: List[Dict[str, str]] = []
    for key, prof in profiles.items():
        label = prof.get("common_name") or key
        opts.append({"label": label, "value": key})
    # sort by label
    opts.sort(key=lambda p: p["label"].lower())
    return opts