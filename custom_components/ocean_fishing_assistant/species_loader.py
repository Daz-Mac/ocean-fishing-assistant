"""Species profile loader for Ocean Fishing Assistant (strict, no fallbacks)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class SpeciesLoader:
    """Load and manage species profiles from the packaged JSON file.

    Strict behaviour: on any load/validation error this loader will raise.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._profiles: Optional[Dict[str, Any]] = None

    async def async_load_profiles(self) -> None:
        """Asynchronously load and validate species_profiles.json.

        Raises RuntimeError on any error so callers can fail loudly.
        """
        json_path = os.path.join(os.path.dirname(__file__), "species_profiles.json")

        def _read_file() -> Dict[str, Any]:
            with open(json_path, "r", encoding="utf-8") as fp:
                return json.load(fp)

        try:
            profiles = await self.hass.async_add_executor_job(_read_file)
        except FileNotFoundError as exc:
            _LOGGER.exception("species_profiles.json not found at %s", json_path)
            raise RuntimeError("species_profiles.json missing") from exc
        except Exception as exc:
            _LOGGER.exception("Failed to read species_profiles.json: %s", exc)
            raise RuntimeError("Failed to read species_profiles.json") from exc

        # Basic schema validation: ensure required top-level keys exist
        if not isinstance(profiles, dict):
            _LOGGER.error("species_profiles.json root element is not a JSON object")
            raise RuntimeError("Invalid species_profiles.json: root not an object")

        if "species" not in profiles or "regions" not in profiles:
            _LOGGER.error("species_profiles.json missing required 'species' or 'regions' keys")
            raise RuntimeError("Invalid species_profiles.json: missing keys")

        if not isinstance(profiles["species"], dict) or not isinstance(profiles["regions"], dict):
            _LOGGER.error("'species' or 'regions' in species_profiles.json have incorrect types")
            raise RuntimeError("Invalid species_profiles.json: incorrect types for keys")

        # Store and log
        version = profiles.get("version", "unknown")
        species_count = len(profiles.get("species", {}))
        _LOGGER.info("Loaded species_profiles.json version %s with %d species", version, species_count)

        self._profiles = profiles

    def _ensure_loaded(self) -> None:
        if self._profiles is None:
            _LOGGER.error("SpeciesLoader used before profiles were loaded")
            raise RuntimeError("Species profiles not loaded")

    def get_species(self, species_id: str) -> Optional[Dict[str, Any]]:
        """Return a copy of the species profile with the given ID, or None."""
        self._ensure_loaded()
        species = self._profiles["species"].get(species_id)
        if species is None:
            return None
        profile = dict(species)
        profile["id"] = species_id
        return profile

    def get_species_by_region(self, region: str) -> List[Dict[str, Any]]:
        """Return list of species available in a given region (copies)."""
        self._ensure_loaded()
        results: List[Dict[str, Any]] = []
        for sid, sdata in self._profiles["species"].items():
            if not isinstance(sdata, dict):
                continue
            available_regions = sdata.get("regions", [])
            if region in available_regions:
                profile = dict(sdata)
                profile["id"] = sid
                results.append(profile)
        return results

    def get_species_by_type(self, species_type: str) -> List[Dict[str, Any]]:
        """Return species by habitat type (e.g. 'ocean' or 'freshwater')."""
        self._ensure_loaded()
        results: List[Dict[str, Any]] = []
        for sid, sdata in self._profiles["species"].items():
            if not isinstance(sdata, dict):
                continue
            habitat = sdata.get("habitat")
            if habitat == species_type:
                profile = dict(sdata)
                profile["id"] = sid
                results.append(profile)
        return results

    def get_all_species(self) -> List[Dict[str, Any]]:
        """Return all species profiles as a list of dicts (copies)."""
        self._ensure_loaded()
        all_species: List[Dict[str, Any]] = []
        for sid, sdata in self._profiles["species"].items():
            if not isinstance(sdata, dict):
                continue
            profile = dict(sdata)
            profile["id"] = sid
            all_species.append(profile)
        return all_species

    def get_regions(self) -> List[Dict[str, Any]]:
        """Return all regions metadata as list of dicts (copies)."""
        self._ensure_loaded()
        regions: List[Dict[str, Any]] = []
        for rid, rdata in self._profiles["regions"].items():
            region_copy: Dict[str, Any] = dict(rdata) if isinstance(rdata, dict) else {"id": rid, "name": str(rid)}
            region_copy.setdefault("id", rid)
            regions.append(region_copy)
        return regions

    def get_regions_by_type(self, region_type: str) -> List[Dict[str, Any]]:
        """Filter regions by habitat type (ocean/freshwater/mixed)."""
        regions = self.get_regions()
        return [r for r in regions if r.get("habitat") == region_type]

    def get_regions_for_species(self, species_id: str) -> List[str]:
        """Return list of region IDs for the given species."""
        species = self.get_species(species_id)
        if not species:
            return []
        return list(species.get("regions", []))

    def get_region_info(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Return region metadata for a specific region, or None."""
        self._ensure_loaded()
        region = self._profiles["regions"].get(region_id)
        if region is None:
            return None
        info = dict(region)
        info.setdefault("id", region_id)
        return info