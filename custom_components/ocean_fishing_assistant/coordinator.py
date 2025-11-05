from datetime import timedelta, datetime
import async_timeout
import logging
import time
from typing import Optional

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store

from .const import STORE_KEY, STORE_VERSION, FETCH_CACHE_TTL, DOMAIN
from .tide_proxy import TideProxy

_LOGGER = logging.getLogger(__name__)


class OFACoordinator(DataUpdateCoordinator):
    def __init__(
        self,
        hass,
        entry_id: str,
        fetcher,
        formatter,
        lat: float,
        lon: float,
        update_interval: int,
        store_enabled: bool = False,
        ttl: int = 3600,
        species: Optional[str] = None,
        units: str = "metric",
        safety_limits: Optional[dict] = None,
    ):
        """Coordinator that fetches, aligns tide, formats data and persists last fetch."""
        super().__init__(
            hass,
            _LOGGER,
            name="ocean_fishing_assistant",
            update_interval=timedelta(seconds=update_interval),
        )
        self.entry_id = entry_id
        self.fetcher = fetcher
        self.formatter = formatter
        self.lat = lat
        self.lon = lon
        # per-entry options
        self.species = species
        self.units = units or "metric"
        # canonicalized safety limits in metric units (e.g. m/s, m, km, s)
        self.safety_limits = safety_limits or {}
        # per-entry store key to avoid cross-entry collisions
        self._store = Store(hass, STORE_VERSION, f"{STORE_KEY}_{entry_id}") if store_enabled else None
        self._ttl = int(ttl)
        # instantiate TideProxy for this sensor coords
        self._tide_proxy = TideProxy(hass, self.lat, self.lon)

    async def _async_update_data(self):
        """Fetch weather (cached), attach tide data, run formatter to precompute forecasts."""
        async with async_timeout.timeout(60):
            # build robust cache access via hass.data
            cache_dict = self.hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
            cache_key = (round(float(self.lat), 4), round(float(self.lon), 4), "hourly", int(5))
            cached = cache_dict.get(cache_key)
            raw = None
            try:
                if cached and (time.time() - float(cached.get("fetched_at", 0))) < FETCH_CACHE_TTL:
                    raw = cached.get("data")
                else:
                    raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)
                    try:
                        cache_dict[cache_key] = {"fetched_at": time.time(), "data": raw}
                    except Exception:
                        _LOGGER.debug("Failed to update in-memory fetch cache", exc_info=True)
            except Exception:
                # fallback to fetching directly if cache logic fails
                raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)

            # Attempt to fetch marine/wave variables and merge them into the hourly payload if available.
            try:
                marine = None
                if hasattr(self.fetcher, "fetch_marine_direct"):
                    _LOGGER.debug("Fetching marine variables via fetch_marine_direct")
                    marine = await self.fetcher.fetch_marine_direct(days=5)
                if isinstance(marine, dict) and isinstance(marine.get("hourly"), dict):
                    marine_hourly = marine.get("hourly", {})
                    # If raw contains hourly arrays, merge keys that are missing. Prefer existing keys in raw.
                    if isinstance(raw, dict) and isinstance(raw.get("hourly"), dict):
                        raw_hourly = raw["hourly"]
                        # Reference times from the primary weather payload
                        ref_time = raw_hourly.get("time") or []
                        ref_len = len(ref_time) if isinstance(ref_time, list) else None
                        marine_time = marine_hourly.get("time") or []
                        # helper to parse ISO timestamps to epoch seconds
                        def _iso_to_ts(s):
                            try:
                                if s is None:
                                    return None
                                ss = str(s)
                                if ss.endswith("Z"):
                                    ss = ss[:-1] + "+00:00"
                                return datetime.fromisoformat(ss).timestamp()
                            except Exception:
                                return None
                        # pre-parse marine timestamps if available
                        marine_ts = [ _iso_to_ts(t) for t in marine_time ] if isinstance(marine_time, list) else []
                        for k, arr in marine_hourly.items():
                            if k == "time":
                                continue
                            if k in raw_hourly:
                                # Prefer existing keys from primary payload
                                continue
                            if not isinstance(arr, list):
                                _LOGGER.debug("Marine key %s is not a list; skipping", k)
                                continue
                            # If we have no reference times, attach as-is
                            if ref_len is None:
                                raw_hourly[k] = arr
                                _LOGGER.debug("Attached marine key %s (no reference time available)", k)
                                continue
                            # If lengths match, attach directly
                            if len(arr) == ref_len:
                                raw_hourly[k] = arr
                                _LOGGER.debug("Attached marine key %s (length matched)", k)
                                continue
                            # Attempt nearest-timestamp alignment if possible
                            if not marine_ts or any(v is None for v in marine_ts):
                                _LOGGER.debug("Marine key %s length mismatch and marine timestamps unavailable or unparsable; skipping", k)
                                continue
                            # parse reference timestamps
                            ref_ts = [ _iso_to_ts(t) for t in ref_time ]
                            if any(v is None for v in ref_ts):
                                _LOGGER.debug("Reference timestamps unparsable; skipping alignment for key %s", k)
                                continue
                            # build aligned array using nearest-neighbor
                            aligned = []
                            for r in ref_ts:
                                # find nearest marine index
                                j = min(range(len(marine_ts)), key=lambda idx: abs(marine_ts[idx] - r))
                                aligned.append(arr[j] if j < len(arr) else None)
                            raw_hourly[k] = aligned
                            _LOGGER.debug("Aligned marine key %s to reference timestamps (src_len=%d ref_len=%d)", k, len(arr), ref_len)
                    else:
                        # Raw had no hourly dict — attach marine hourly as the hourly payload
                        raw.setdefault("hourly", {}).update(marine_hourly)
                        _LOGGER.debug("Attached marine hourly payload as primary hourly data")
                        # Also, if top-level time/timestamps missing, try convenience keys
                        if "time" not in raw and "time" in marine_hourly:
                            raw["time"] = marine_hourly.get("time")
            except Exception:
                _LOGGER.debug("Marine merge failed; continuing without marine data", exc_info=True)

            # attempt to align tide to weather timestamps if available
            timestamps = None
            if isinstance(raw, dict) and isinstance(raw.get("hourly"), dict):
                timestamps = raw.get("hourly", {}).get("time") or raw.get("timestamps") or raw.get("time") or []
            else:
                timestamps = raw.get("timestamps") or raw.get("time") or []

            if timestamps:
                try:
                    tide = await self._tide_proxy.get_tide_for_timestamps(timestamps)
                    # Keep a top-level tide dict for the formatter to consume
                    raw.setdefault("tide", {}).update(tide)
                    # Backwards-compatible convenience keys
                    if "tide_height_m" not in raw and tide.get("tide_height_m") is not None:
                        raw["tide_height_m"] = tide.get("tide_height_m")
                    if "tide_phase" not in raw and tide.get("tide_phase") is not None:
                        raw["tide_phase"] = tide.get("tide_phase")
                    if "tide_strength" not in raw and tide.get("tide_strength") is not None:
                        raw["tide_strength"] = tide.get("tide_strength")
                except Exception:
                    _LOGGER.debug("TideProxy failed; continuing without tide", exc_info=True)

            # Attach convenience 'current' snapshot if the fetcher can provide it
            try:
                if hasattr(self.fetcher, "get_weather_data"):
                    current = await self.fetcher.get_weather_data()
                    if current:
                        raw["current"] = current
            except Exception:
                _LOGGER.debug("Failed to get current snapshot from fetcher", exc_info=True)

            # IMPORTANT: pass per-entry species, units and safety_limits into the formatter
            data = self.formatter.validate(
                raw,
                species_profile=self.species,
                units=self.units,
                safety_limits=self.safety_limits,
            )

            # persist if store enabled (wrap with timestamp for TTL checks)
            if self._store:
                try:
                    await self._store.async_save({"fetched_at": time.time(), "data": data})
                except Exception:
                    _LOGGER.debug("Failed to persist fetch to store", exc_info=True)

            return data