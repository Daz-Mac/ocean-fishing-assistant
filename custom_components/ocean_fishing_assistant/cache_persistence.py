"""Persistent cache save/load for Ocean Fishing Assistant.

Centralizes Store initialization, serialization, and periodic persistence
of in-memory caches across HA restarts.
"""
import logging
from datetime import timedelta, timezone
from typing import Any, Dict

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.helpers.storage import Store
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    FETCH_CACHE_TTL,
    SHARED_TIDE_CACHE_KEY,
    SHARED_TIDE_INFLIGHT_KEY,
    WEATHER_FETCHER_CACHE_TTL_DEFAULT,
)

_LOGGER = logging.getLogger(__name__)

_STORE_VERSION = 1
_STORE_KEY = "ocean_fishing_assistant_persisted_cache_v1"
_STORE_COORDINATOR_FETCH_KEY = "coordinator_fetch_cache"
_STORE_WEATHER_FETCH_KEY = "weather_fetch_cache"
_STORE_TIDE_CACHE_KEY = "tide_api_cache"
_STORE_META_KEY = "_meta"
_PERSIST_SAVE_INTERVAL = timedelta(minutes=5)


def _serialize_coord_cache_key(key: Any) -> str:
    try:
        if isinstance(key, (list, tuple)) and len(key) == 4:
            lat, lon, mode, days = key
            return f"fetch|{float(lat)}|{float(lon)}|{str(mode)}|{int(days)}"
    except Exception:
        pass
    return str(key)


def _deserialize_coord_cache_key(k: str):
    try:
        if k.startswith("fetch|"):
            _, lat_s, lon_s, mode_s, days_s = k.split("|", 4)
            return (float(lat_s), float(lon_s), mode_s, int(days_s))
    except Exception:
        pass
    return k


async def async_load_and_setup_persistence(hass, domain_store):
    """Initialize Store, load persisted caches, and register save listeners.

    Idempotent — only performs full setup on first call. Subsequent calls
    ensure required in-memory cache dicts exist.
    """
    if domain_store.get("_persist_initialized"):
        domain_store.setdefault("fetch_cache", {})
        domain_store.setdefault(SHARED_TIDE_CACHE_KEY, {})
        domain_store.setdefault(SHARED_TIDE_INFLIGHT_KEY, {})
        hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {})
        return

    store = Store(hass, _STORE_VERSION, _STORE_KEY)
    try:
        persisted = await store.async_load() or {}
    except Exception:
        _LOGGER.exception("Failed to load persisted cache store; continuing with empty caches")
        persisted = {}

    meta = persisted.get(_STORE_META_KEY)
    if not isinstance(meta, dict) or int(meta.get("version", 0)) != _STORE_VERSION:
        _LOGGER.debug(
            "Persisted cache _meta missing or version mismatch (found=%s expected=%s)",
            meta, _STORE_VERSION,
        )

    # Coordinator shared fetch cache
    raw_coord_cache = persisted.get(_STORE_COORDINATOR_FETCH_KEY) or {}
    coord_cache: Dict[Any, Any] = {}
    try:
        now_ts = int(dt_util.now().timestamp())
        for k_str, v in raw_coord_cache.items():
            try:
                tup_key = _deserialize_coord_cache_key(k_str)
                fetched_at = float(v.get("fetched_at", 0))
                if (now_ts - int(fetched_at)) < int(FETCH_CACHE_TTL):
                    coord_cache[tup_key] = v
            except Exception:
                _LOGGER.debug("Skipping malformed coordinator cache entry on load: %s", k_str)
    except Exception:
        _LOGGER.exception("Failed to reconstruct persisted coordinator fetch cache")
    domain_store.setdefault("fetch_cache", coord_cache)

    # Tide shared cache
    raw_tide_cache = persisted.get(_STORE_TIDE_CACHE_KEY) or {}
    tide_cache: Dict[str, Any] = {}
    try:
        now_ts = int(dt_util.now().timestamp())
        for k_str, v in raw_tide_cache.items():
            try:
                expires = int(v.get("expires", 0))
                if expires >= now_ts:
                    tide_cache[k_str] = v
            except Exception:
                _LOGGER.debug("Skipping malformed tide cache entry on load: %s", k_str)
    except Exception:
        _LOGGER.exception("Failed to reconstruct persisted tide cache")
    domain_store.setdefault(SHARED_TIDE_CACHE_KEY, tide_cache)
    domain_store.setdefault(SHARED_TIDE_INFLIGHT_KEY, {})

    # WeatherFetcher cache
    raw_weather_cache = persisted.get(_STORE_WEATHER_FETCH_KEY) or {}
    weather_cache: Dict[str, Any] = {}
    try:
        for k_str, v in raw_weather_cache.items():
            try:
                entry_time = v.get("time")
                parsed_dt = None
                if entry_time is not None:
                    try:
                        tnum = float(entry_time)
                        if tnum > 1e12:
                            tnum = tnum / 1000.0
                        parsed_dt = dt_util.utc_from_timestamp(tnum)
                    except Exception:
                        try:
                            parsed = dt_util.parse_datetime(str(entry_time))
                            if parsed and parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            parsed_dt = parsed
                        except Exception:
                            parsed_dt = None

                if parsed_dt:
                    now_ts = int(dt_util.now().timestamp())
                    entry_ts = int(parsed_dt.timestamp())
                    if (now_ts - entry_ts) < int(WEATHER_FETCHER_CACHE_TTL_DEFAULT):
                        weather_cache[k_str] = {"data": v.get("data"), "time": parsed_dt}
            except Exception:
                _LOGGER.debug("Skipping malformed weather cache entry on load: %s", k_str)
    except Exception:
        _LOGGER.exception("Failed to reconstruct persisted weather fetch cache")
    hass.data.setdefault("ocean_fishing_assistant_fetch_cache", weather_cache)

    domain_store["_persist_store"] = store
    domain_store["_persist_initialized"] = True

    async def _do_persist_save():
        try:
            s = domain_store.get("_persist_store") or store
            to_save: Dict[str, Any] = {}
            to_save[_STORE_META_KEY] = {"version": _STORE_VERSION, "saved_at": int(dt_util.now().timestamp())}

            coord_cache_local = domain_store.get("fetch_cache", {}) or {}
            coord_serialized = {}
            for k, v in coord_cache_local.items():
                try:
                    kstr = _serialize_coord_cache_key(k)
                    coord_serialized[kstr] = v
                except Exception:
                    _LOGGER.debug("Skipping non-serializable coordinator fetch_cache key on save: %r", k)
            to_save[_STORE_COORDINATOR_FETCH_KEY] = coord_serialized

            tide_cache_local = domain_store.get(SHARED_TIDE_CACHE_KEY, {}) or {}
            to_save[_STORE_TIDE_CACHE_KEY] = tide_cache_local

            weather_cache_local = hass.data.get("ocean_fishing_assistant_fetch_cache", {}) or {}
            weather_serialized = {}
            for k, v in weather_cache_local.items():
                try:
                    tval = v.get("time")
                    t_out = None
                    if tval is not None:
                        try:
                            t_out = int(dt_util.as_timestamp(tval))
                        except Exception:
                            try:
                                t_out = int(getattr(tval, "timestamp")())
                            except Exception:
                                t_out = None
                    weather_serialized[k] = {"data": v.get("data"), "time": t_out}
                except Exception:
                    _LOGGER.debug("Skipping non-serializable weather cache entry on save: %s", k)
            to_save[_STORE_WEATHER_FETCH_KEY] = weather_serialized

            await s.async_save(to_save)
            _LOGGER.debug(
                "Persisted caches: coord=%d tide=%d weather=%d",
                len(coord_serialized),
                len(tide_cache_local),
                len(weather_serialized),
            )
        except Exception:
            _LOGGER.exception("Failed to persist caches (periodic/stop)")

    async def _save_on_stop(event):
        await _do_persist_save()

    try:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _save_on_stop)
    except Exception:
        _LOGGER.exception("Failed to register stop listener for cache persistence")

    try:
        async def _periodic_save(now):
            await _do_persist_save()

        periodic_unsub = async_track_time_interval(hass, _periodic_save, _PERSIST_SAVE_INTERVAL)
        domain_store["_persist_periodic_unsub"] = periodic_unsub
    except Exception:
        _LOGGER.exception("Failed to register periodic cache persister")
