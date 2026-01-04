# custom_components/ocean_fishing_assistant/tide_proxy.py
from __future__ import annotations
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncio
import json

from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

# Skyfield (loaded lazily for moon/sun computations)
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield.framelib import ecliptic_frame
import skyfield

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_GRID_SECONDS_DEFAULT = 60  # resolution for nearest-sample matching (seconds)
_NEGATIVE_TTL_DEFAULT = 360  # seconds for negative-cache entries on failure

# Default World Tides base (configurable via const if needed)
from .const import (
    WORLD_TIDES_API_BASE,
    DOMAIN,
    COORD_ROUND_DECIMALS,
    TIDE_PROXY_TTL_DEFAULT,
)

# numeric tolerances
EPS_DERIV = 1e-10

# Minimal date formatting helper
def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _to_epoch_seconds(ts: Any) -> int:
    if isinstance(ts, (int, float)):
        v = float(ts)
        # Sometimes timestamps are milliseconds
        if v > 1e12:
            v = v / 1000.0
        return int(v)
    if isinstance(ts, str):
        s = ts.strip()
        try:
            # numeric-string?
            if s.replace(".", "", 1).isdigit():
                v = float(s)
                if v > 1e12:
                    v = v / 1000.0
                return int(v)
        except Exception:
            pass
        try:
            if s.endswith("Z"):
                s2 = s.replace("Z", "+00:00")
            else:
                s2 = s
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return int(dt.timestamp())
        except Exception:
            raise ValueError(f"Unrecognized timestamp string: {ts}")
    if isinstance(ts, datetime):
        return int(ts.astimezone(timezone.utc).timestamp())
    raise ValueError(f"Unsupported timestamp type: {type(ts)}")


class TideProxy:
    """
    TideProxy backed by World Tides API.

    Strict behavior:
      - World Tides API key must be provided via `api_key` at construction (no fallbacks).
      - get_tide_for_timestamps must return arrays exactly aligned with the provided timestamps (lengths must match).
      - If the World Tides API does not return expected data, a RuntimeError is raised.

    Caching / inflight dedupe:
      - Shared in-memory cache stored at hass.data[DOMAIN]["tide_api_cache"]
      - Inflight map stored at hass.data[DOMAIN]["tide_api_inflight"]
      - Cache keys normalized by rounded coords and canonical time buckets so nearby requests coalesce.
      - Failed requests are negative-cached for `negative_ttl` seconds to avoid hammering API.
    """

    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        ttl: int = TIDE_PROXY_TTL_DEFAULT,
        *,
        phase_offset_hours: float = 0.0,
        api_key: Optional[str] = None,
        worldtides_base: Optional[str] = None,
        negative_ttl: Optional[int] = None,
    ):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._negative_ttl = int(negative_ttl if negative_ttl is not None else _NEGATIVE_TTL_DEFAULT)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

        self._phase_offset_hours = float(phase_offset_hours)

        # API key (strict: must be present)
        if not api_key:
            raise RuntimeError("World Tides API key is required for TideProxy (strict)")
        self._api_key = str(api_key).strip()

        self._base = worldtides_base or WORLD_TIDES_API_BASE

        # Skyfield loader for moon phase and dawn/dusk computations; loaded lazily
        try:
            data_dir = hass.config.path("custom_components", "ocean_fishing_assistant", "data")
        except Exception:
            from homeassistant.const import CONFIG_DIR  # type: ignore
            data_dir = os.path.join(CONFIG_DIR, "custom_components", "ocean_fishing_assistant", "data")
        os.makedirs(data_dir, exist_ok=True)
        self._loader = Loader(data_dir)
        self._sf_ts = None
        self._sf_eph = None
        self._sf_wgs = None
        self._sf_almanac = None
        self._load_lock = asyncio.Lock()

        _LOGGER.debug(
            "TideProxy initialized lat=%s lon=%s ttl=%s negative_ttl=%s phase_offset_hours=%.3f world_base=%s",
            self.latitude,
            self.longitude,
            self._ttl,
            self._negative_ttl,
            self._phase_offset_hours,
            self._base,
        )

    # -----------------------
    # Shared cache / inflight helpers
    # -----------------------
    def _shared_store(self) -> Dict[str, Dict]:
        """
        Return the integration-level store dict under hass.data[DOMAIN].
        Creates structure if missing.
        """
        store = self.hass.data.setdefault(DOMAIN, {})
        # initialize sub-structures
        store.setdefault("tide_api_cache", {})
        store.setdefault("tide_api_inflight", {})
        return store

    def _rounded_coords(self) -> Tuple[float, float]:
        return (
            round(self.latitude, COORD_ROUND_DECIMALS),
            round(self.longitude, COORD_ROUND_DECIMALS),
        )

    def _time_bucket(self, dt: datetime, bucket_seconds: int = 300) -> int:
        # bucket_seconds default 5 minutes: canonicalize start/end to reduce unique keys
        return int(math.floor(dt.timestamp() / float(bucket_seconds)) * bucket_seconds)

    def _make_cache_key(self, endpoint: str, start_dt: datetime, end_dt: datetime, *, step_seconds: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> str:
        lat_r, lon_r = self._rounded_coords()
        start_b = self._time_bucket(start_dt, bucket_seconds=300)
        end_b = self._time_bucket(end_dt, bucket_seconds=300)
        parts = [endpoint, f"lat={lat_r}", f"lon={lon_r}", f"start={start_b}", f"end={end_b}"]
        if step_seconds:
            parts.append(f"step={int(step_seconds)}")
        if extra:
            # keep deterministic ordering
            for k in sorted(extra.keys()):
                parts.append(f"{k}={extra[k]}")
        return "|".join(parts)

    def _get_cached(self, key: str) -> Optional[Any]:
        store = self._shared_store()
        cache = store.get("tide_api_cache", {})
        entry = cache.get(key)
        if not entry:
            _LOGGER.debug("Tide cache miss (no entry) key=%s", key)
            return None
        expires_at = entry.get("expires", 0)
        now_ts = int(dt_util.now().timestamp())
        if expires_at < now_ts:
            _LOGGER.debug("Tide cache entry expired: key=%s", key)
            # remove expired entry
            try:
                del cache[key]
            except Exception:
                pass
            return None

        # If entry is an error (negative cache), raise the recorded error immediately
        if entry.get("error"):
            msg = entry.get("message", "previous tide request failed")
            _LOGGER.debug("Tide negative-cache hit key=%s message=%s", key, msg)
            raise RuntimeError(f"Tide fetch previously failed: {msg}")

        _LOGGER.debug("Tide cache hit key=%s", key)
        return entry.get("data")

    def _set_cached(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        store = self._shared_store()
        cache = store.setdefault("tide_api_cache", {})
        ttl_use = int(ttl if ttl is not None else self._ttl)
        expires_at = int(dt_util.now().timestamp()) + max(1, ttl_use)
        cache[key] = {"expires": expires_at, "data": data}
        _LOGGER.debug("Tide cache set key=%s expires_in=%s", key, ttl_use)

    def _set_cached_error(self, key: str, exc: Exception, ttl: Optional[int] = None) -> None:
        """
        Write a negative cache entry for `key` recording the error message for `ttl` seconds.
        """
        store = self._shared_store()
        cache = store.setdefault("tide_api_cache", {})
        ttl_use = int(ttl if ttl is not None else self._negative_ttl)
        expires_at = int(dt_util.now().timestamp()) + max(1, ttl_use)
        # store stringified message only (do not store exception object)
        msg = str(exc)
        cache[key] = {"expires": expires_at, "error": True, "message": msg}
        _LOGGER.debug("Tide negative-cache set key=%s expires_in=%s message=%s", key, ttl_use, msg)

    async def _await_inflight_or_run(self, key: str, coro_func):
        """
        If a request for `key` is already in-flight, await its future.
        Otherwise, register a new future, run coro_func() to produce result,
        set result on future and return it.

        coro_func should be a coroutine function (callable) that, when awaited, returns the result.
        """
        store = self._shared_store()
        inflight = store.setdefault("tide_api_inflight", {})

        loop = asyncio.get_running_loop()
        existing = inflight.get(key)
        if existing:
            _LOGGER.debug("Awaiting inflight tide request key=%s", key)
            try:
                return await asyncio.shield(existing)
            finally:
                # do not remove here; the creator will remove
                pass

        fut: asyncio.Future = loop.create_future()
        inflight[key] = fut
        try:
            _LOGGER.debug("Starting network request for tide key=%s", key)
            result = await coro_func()
            # store into cache by caller if desired, but we set it here to ensure consistent caching
            if not fut.done():
                fut.set_result(result)
            return result
        except Exception as exc:
            _LOGGER.debug("Tide request failed for key=%s exc=%s", key, exc)
            # set a negative cache entry so subsequent immediate callers see the failure for a short window
            try:
                self._set_cached_error(key, exc, ttl=self._negative_ttl)
            except Exception:
                _LOGGER.debug("Failed to write negative cache for key=%s", key, exc_info=True)
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            # remove inflight entry
            try:
                inflight.pop(key, None)
            except Exception:
                pass

    # -----------------------
    # Fetch wrappers that use shared cache + inflight dedupe
    # -----------------------
    async def _fetch_world_tides_heights(self, start_dt: datetime, end_dt: datetime, step_seconds: int = 60) -> List[Dict[str, Any]]:
        """
        Request World Tides heights for the interval [start_dt, end_dt] with approximate step (seconds).
        Uses shared cache and inflight de-duplication.
        Returns a list of dicts with keys: 'time' (datetime UTC) and 'height' (meters).
        Raises RuntimeError on API problems.
        """
        # Build canonical cache key
        key = self._make_cache_key("heights", start_dt, end_dt, step_seconds=step_seconds)
        cached = None
        try:
            cached = self._get_cached(key)
        except RuntimeError:
            # negative-cache hit -> raise immediately
            raise
        if cached is not None:
            # cached stored with ISO times; convert to datetime objects
            out: List[Dict[str, Any]] = []
            for item in cached:
                try:
                    ts = item.get("time")
                    if isinstance(ts, str):
                        s = ts
                        if s.endswith("Z"):
                            s2 = s.replace("Z", "+00:00")
                        else:
                            s2 = s
                        dt = datetime.fromisoformat(s2)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                    elif isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    else:
                        dt = ts
                    out.append({"time": dt, "height": float(item.get("height"))})
                except Exception:
                    # skip malformed cached entry (treat as cache miss)
                    _LOGGER.debug("Malformed cached height entry for key=%s: %r", key, item)
                    out = None
                    break
            if out is not None:
                return out

        # define the actual network call as a coroutine
        async def _do_fetch():
            # Build request window: world tides expects start date as ISO (UTC)
            start_iso = _iso_z(start_dt)
            # length in days (float) — ensure cover whole interval
            seconds = max(1, int((end_dt - start_dt).total_seconds()))
            length_days = max(1, math.ceil(seconds / 86400.0))

            session = async_get_clientsession(self.hass)

            params = {
                "lat": str(self.latitude),
                "lon": str(self.longitude),
                "start": start_iso,
                "length": str(length_days),
                "key": self._api_key,
                # request minute resolution where supported
                "interval": str(max(60, int(step_seconds))),
            }

            url = f"{self._base}/heights"
            _LOGGER.debug("WorldTides heights request: %s params=%s", url, params)
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _LOGGER.error("WorldTides heights request failed status=%s body=%s", resp.status, body)
                        raise RuntimeError(f"WorldTides heights request failed status={resp.status}")
                    j = await resp.json()
            except Exception as exc:
                _LOGGER.exception("WorldTides heights request exception: %s", exc)
                raise RuntimeError(f"Failed to fetch World Tides heights: {exc}") from exc

            # Support a few possible keys ('heights', 'predictions', 'data')
            raw_data = None
            for k in ("heights", "predictions", "data"):
                raw_data = j.get(k)
                if raw_data:
                    break
            if not raw_data:
                _LOGGER.error("WorldTides heights returned unexpected payload: %s", j)
                raise RuntimeError("WorldTides heights returned unexpected payload (missing heights/predictions)")

            out_cacheable: List[Dict[str, Any]] = []
            out_parsed: List[Dict[str, Any]] = []
            for item in raw_data:
                # item could contain 'date'/'time' and 'height' or 't' and 'v'
                ts = item.get("date") or item.get("time") or item.get("t")
                h = item.get("height") or item.get("v") or item.get("value")
                if ts is None or h is None:
                    continue
                # Normalize timestamp: world tides often returns ISO in zulu or local format — parse defensively
                try:
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                        ts_out = _iso_z(dt)
                    else:
                        s = str(ts)
                        if s.endswith("Z"):
                            s2 = s.replace("Z", "+00:00")
                        else:
                            s2 = s
                        dt = datetime.fromisoformat(s2)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        ts_out = _iso_z(dt)
                except Exception:
                    _LOGGER.debug("Failed to parse WorldTides time value %r", ts)
                    continue
                try:
                    height_m = float(h)
                except Exception:
                    _LOGGER.debug("Failed to parse WorldTides height value %r", h)
                    continue
                out_parsed.append({"time": dt, "height": height_m})
                out_cacheable.append({"time": ts_out, "height": height_m})

            out_parsed.sort(key=lambda x: x["time"].timestamp())
            out_cacheable.sort(key=lambda x: x["time"])
            if not out_parsed:
                raise RuntimeError("WorldTides heights response contained no usable samples")

            # write cache (cache TTL uses proxy ttl)
            try:
                self._set_cached(key, out_cacheable, ttl=self._ttl)
            except Exception:
                _LOGGER.debug("Failed to set tide heights cache for key=%s", key)
            return out_parsed

        # Use inflight / dedupe wrapper
        try:
            return await self._await_inflight_or_run(key, _do_fetch)
        except Exception:
            raise

    async def _fetch_world_tides_extremes(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """
        Request World Tides extremes (high/low) for [start_dt, end_dt].
        Uses shared cache and inflight de-duplication.
        Returns list of dicts with keys like 'time' (datetime UTC) and 'height' and 'type' ('High'/'Low') where available.
        """
        key = self._make_cache_key("extremes", start_dt, end_dt, step_seconds=None)
        cached = None
        try:
            cached = self._get_cached(key)
        except RuntimeError:
            # negative-cache hit -> raise immediately
            raise
        if cached is not None:
            out: List[Dict[str, Any]] = []
            for item in cached:
                try:
                    ts = item.get("time")
                    if isinstance(ts, str):
                        s = ts
                        if s.endswith("Z"):
                            s2 = s.replace("Z", "+00:00")
                        else:
                            s2 = s
                        dt = datetime.fromisoformat(s2)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                    elif isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    else:
                        dt = ts
                    out.append({"time": dt, "height": float(item.get("height")), "type": item.get("type")})
                except Exception:
                    _LOGGER.debug("Malformed cached extremes entry for key=%s: %r", key, item)
                    out = None
                    break
            if out is not None:
                return out

        async def _do_fetch():
            start_iso = _iso_z(start_dt)
            seconds = max(1, int((end_dt - start_dt).total_seconds()))
            length_days = max(1, math.ceil(seconds / 86400.0))

            session = async_get_clientsession(self.hass)
            params = {
                "lat": str(self.latitude),
                "lon": str(self.longitude),
                "start": start_iso,
                "length": str(length_days),
                "key": self._api_key,
            }
            url = f"{self._base}/extremes"
            _LOGGER.debug("WorldTides extremes request: %s params=%s", url, params)
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _LOGGER.error("WorldTides extremes request failed status=%s body=%s", resp.status, body)
                        raise RuntimeError(f"WorldTides extremes request failed status={resp.status}")
                    j = await resp.json()
            except Exception as exc:
                _LOGGER.exception("WorldTides extremes request exception: %s", exc)
                raise RuntimeError(f"Failed to fetch World Tides extremes: {exc}") from exc

            raw_data = None
            for k in ("extremes", "data", "predictions"):
                raw_data = j.get(k)
                if raw_data:
                    break
            if not raw_data:
                _LOGGER.debug("WorldTides extremes response keys: %s", list(j.keys()))
                raise RuntimeError("WorldTides extremes returned unexpected payload (missing extremes)")

            out_cacheable: List[Dict[str, Any]] = []
            out_parsed: List[Dict[str, Any]] = []
            for item in raw_data:
                ts = item.get("date") or item.get("time") or item.get("t")
                h = item.get("height") or item.get("v") or item.get("value")
                typ = item.get("type") or item.get("extreme") or None
                if ts is None or h is None:
                    continue
                try:
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                        ts_out = _iso_z(dt)
                    else:
                        s = str(ts)
                        if s.endswith("Z"):
                            s2 = s.replace("Z", "+00:00")
                        else:
                            s2 = s
                        dt = datetime.fromisoformat(s2)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        ts_out = _iso_z(dt)
                except Exception:
                    _LOGGER.debug("Failed to parse WorldTides extreme time value %r", ts)
                    continue
                try:
                    height_m = float(h)
                except Exception:
                    _LOGGER.debug("Failed to parse WorldTides extreme height value %r", h)
                    continue
                out_parsed.append({"time": dt, "height": height_m, "type": typ})
                out_cacheable.append({"time": ts_out, "height": height_m, "type": typ})

            out_parsed.sort(key=lambda x: x["time"].timestamp())
            out_cacheable.sort(key=lambda x: x["time"])
            # store into cache
            try:
                self._set_cached(key, out_cacheable, ttl=self._ttl)
            except Exception:
                _LOGGER.debug("Failed to set tide extremes cache for key=%s", key)
            return out_parsed

        try:
            return await self._await_inflight_or_run(key, _do_fetch)
        except Exception:
            raise

    # -----------------------
    # High-level API (unchanged except using shared cache-backed fetchers)
    # -----------------------
    async def get_tide_for_timestamps(self, timestamps: Sequence[Any], *, location_tz: str) -> Dict[str, Any]:
        """
        Fetch tide predictions aligned to the provided timestamps using World Tides API.

        Returns a dict with:
          - 'timestamps': list[ISOZ]
          - 'tide_phase': list[str] per timestamp (rising/falling/high/low/flat)
          - 'tide_phase_name': list[str] friendly names ("High Tide"/"Low Tide"/...)
          - 'moon_phase': list[float or None] aligned numeric moon phase (0..1) where possible
          - 'tide_strength': float 0..1 (normalized local amplitude estimate)
          - 'next_high'/'next_low': dict or None, each {'timestamp': ISOZ}
          - 'confidence': "worldtides_api"
          - 'source': "worldtides_api"
        """
        if not location_tz:
            raise ValueError("location_tz is required (strict)")

        if not timestamps:
            return {
                "timestamps": [],
                "tide_phase": [],
                "moon_phase": [],
                "tide_strength": 0.0,
                "confidence": "no_timestamps",
                "source": "worldtides_api",
                "next_high": None,
                "next_low": None,
            }

        # Normalize timestamps into UTC datetime objects
        dt_objs: List[datetime] = []
        for ts in timestamps:
            if isinstance(ts, datetime):
                dt = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                dt_objs.append(dt)
                continue
            try:
                epoch = None
                if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.strip().replace(".", "", 1).isdigit()):
                    v = float(ts)
                    if v > 1e12:
                        v = v / 1000.0
                    epoch = int(v)
                    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    dt_objs.append(dt)
                    continue
            except Exception:
                pass
            parsed = dt_util.parse_datetime(str(ts))
            if parsed is None:
                try:
                    v = float(str(ts))
                    if v > 1e12:
                        v = v / 1000.0
                    parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception:
                    parsed = None
            if parsed is None:
                _LOGGER.error("Unable to parse timestamp: %s", ts)
                raise ValueError("Unable to parse timestamp: %s" % ts)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)

        # TTL/caching: if we recently computed same timestamps, return cached result
        now = dt_util.now().astimezone(timezone.utc)
        try:
            new_keys = [_to_epoch_seconds(t) for t in timestamps]
        except Exception:
            new_keys = [int(dt.timestamp()) for dt in dt_objs]

        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached_keys = self._cache.get("timestamps")
            if cached_keys is not None and cached_keys == new_keys:
                return self._cache["raw_tide"]

        # Determine fetch window: include a margin of 1 hour on each side
        start_dt = min(dt_objs) - timedelta(hours=1)
        end_dt = max(dt_objs) + timedelta(hours=1)

        # Ask World Tides for heights at reasonably fine resolution (1 minute)
        try:
            heights_samples = await self._fetch_world_tides_heights(start_dt, end_dt, step_seconds=_GRID_SECONDS_DEFAULT)
        except Exception as exc:
            _LOGGER.exception("WorldTides heights fetch failed: %s", exc)
            raise RuntimeError(f"WorldTides heights fetch failed: {exc}") from exc

        # Build time-indexed list for nearest-sample lookups
        times_arr = [float(x["time"].timestamp()) for x in heights_samples]
        heights_arr = [float(x["height"]) for x in heights_samples]

        # Helper: find nearest height sample index for epoch t (binary search)
        import bisect

        def _nearest_sample_value(epoch_ts: float) -> Tuple[float, int]:
            i = bisect.bisect_left(times_arr, epoch_ts)
            if i == 0:
                return heights_arr[0], 0
            if i >= len(times_arr):
                return heights_arr[-1], len(times_arr) - 1
            before_t = times_arr[i - 1]
            after_t = times_arr[i]
            if abs(epoch_ts - before_t) <= abs(after_t - epoch_ts):
                return heights_arr[i - 1], i - 1
            return heights_arr[i], i

        tidal_heights_for_timestamps: List[float] = []
        sample_indices_used: List[int] = []
        for dt in dt_objs:
            epoch = float(dt.timestamp())
            h, idx = _nearest_sample_value(epoch)
            tidal_heights_for_timestamps.append(float(h))
            sample_indices_used.append(int(idx))

        # Determine tide phase for each timestamp using numeric derivative around nearest sample
        tide_phase_list: List[str] = []
        tide_phase_name_list: List[str] = []
        n_samples = len(heights_arr)
        for idx in sample_indices_used:
            try:
                # numeric derivative: use neighbor samples where possible
                if n_samples == 1:
                    d = 0.0
                else:
                    # choose indices safely
                    i0 = max(0, idx - 1)
                    i1 = min(n_samples - 1, idx + 1)
                    t0 = times_arr[i0]
                    t1 = times_arr[i1]
                    y0 = heights_arr[i0]
                    y1 = heights_arr[i1]
                    if t1 == t0:
                        d = 0.0
                    else:
                        d = (y1 - y0) / (t1 - t0)
                if abs(d) < 1e-6:
                    phase = "flat"
                elif d > 0.0:
                    phase = "rising"
                else:
                    phase = "falling"
                tide_phase_list.append(phase)
                if phase == "rising":
                    tide_phase_name_list.append("Rising")
                elif phase == "falling":
                    tide_phase_name_list.append("Falling")
                else:
                    tide_phase_name_list.append("Flat")
            except Exception:
                tide_phase_list.append("flat")
                tide_phase_name_list.append("Flat")

        # Find extrema (local highs and lows) from heights_samples after now
        highs: List[Tuple[float, float]] = []
        lows: List[Tuple[float, float]] = []
        try:
            for i in range(1, len(heights_arr) - 1):
                prev_h = heights_arr[i - 1]
                cur_h = heights_arr[i]
                next_h = heights_arr[i + 1]
                t_epoch = float(times_arr[i])
                if cur_h > prev_h and cur_h > next_h:
                    highs.append((t_epoch, cur_h))
                elif cur_h < prev_h and cur_h < next_h:
                    lows.append((t_epoch, cur_h))
        except Exception:
            _LOGGER.exception("Failed to identify extrema from WorldTides heights")

        now_ts = float(now.timestamp())
        next_high_obj = None
        next_low_obj = None
        for t, h in highs:
            if t >= now_ts:
                next_high_obj = {"timestamp": datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")}
                break
        for t, h in lows:
            if t >= now_ts:
                next_low_obj = {"timestamp": datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")}
                break

        # If extremes endpoint available, fetch and prefer it for provenance
        try:
            extremes = await self._fetch_world_tides_extremes(start_dt, end_dt)
            # convert extremes entries into next_high/next_low (prefer first future ones)
            if extremes:
                # find first high/low after now
                fut_high = next((e for e in extremes if float(e["time"].timestamp()) >= now_ts and (str(e.get("type", "")).lower().startswith("h") or str(e.get("type", "")).lower() == "high")), None)
                fut_low = next((e for e in extremes if float(e["time"].timestamp()) >= now_ts and (str(e.get("type", "")).lower().startswith("l") or str(e.get("type", "")).lower() == "low")), None)
                if fut_high:
                    next_high_obj = {"timestamp": fut_high["time"].isoformat().replace("+00:00", "Z")}
                if fut_low:
                    next_low_obj = {"timestamp": fut_low["time"].isoformat().replace("+00:00", "Z")}
        except Exception:
            # non-fatal: we already have next_high_obj/next_low_obj from heights interpolation above
            _LOGGER.debug("WorldTides extremes fetch failed or not available; continuing with height-derived extrema", exc_info=True)

        # Compute simple tide_strength estimate (normalized amplitude over sample window)
        try:
            amp = (max(heights_arr) - min(heights_arr)) / 2.0
            # Normalize to some plausible amplitude range (domain-specific tuning)
            # Use 0..5m typical range -> clamp to [0,1]
            norm = max(0.0, min(1.0, amp / 2.5))
            tide_strength_value = float(norm)
        except Exception:
            tide_strength_value = 0.5

        # Compute moon phases for the timestamps (attempt skyfield but do not fail if unavailable)
        moon_phases: List[Optional[float]] = []
        try:
            await self._ensure_loaded()
            sf_ts = self._sf_ts
            sf_eph = self._sf_eph
            earth = sf_eph["earth"]
            sun_obj = sf_eph["sun"]
            moon_obj = sf_eph["moon"]
            times_list = [sf_ts.from_datetime(dt) for dt in dt_objs]
            for t in times_list:
                try:
                    sun_app = earth.at(t).observe(sun_obj).apparent()
                    moon_app = earth.at(t).observe(moon_obj).apparent()
                    sun_ecl = sun_app.frame_latlon(ecliptic_frame)
                    moon_ecl = moon_app.frame_latlon(ecliptic_frame)
                    lon_sun = float(sun_ecl[1].degrees)
                    lon_moon = float(moon_ecl[1].degrees)
                    diff = (lon_moon - lon_sun) % 360.0
                    moon_phases.append(diff / 360.0)
                except Exception:
                    moon_phases.append(None)
        except Exception:
            # If Skyfield unavailable, fall back to None for moon phase (we keep strictness on tide heights only)
            moon_phases = [None] * len(dt_objs)

        # Build canonical arrays of ISOZ timestamps
        ts_isoz = [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs]

        raw_tide: Dict[str, Any] = {
            "timestamps": ts_isoz,
            "moon_phase": moon_phases,
            "tide_heights_m": tidal_heights_for_timestamps,
            "tide_phase": tide_phase_list,
            "tide_phase_name": tide_phase_name_list,
            "tide_strength": float(round(tide_strength_value, 3)),
            "confidence": "worldtides_api",
            "source": "worldtides_api",
            "_helpers": {
                "samples_used": len(heights_samples),
                "heights_window_start": _iso_z(start_dt),
                "heights_window_end": _iso_z(end_dt),
                "phase_offset_hours": float(self._phase_offset_hours),
            },
            "next_high": next_high_obj,
            "next_low": next_low_obj,
        }

        # Cache and return (per-instance exact-cache)
        try:
            self._cache = {"timestamps": new_keys, "raw_tide": raw_tide, "version": 1}
        except Exception:
            self._cache = {"timestamps": new_keys, "raw_tide": raw_tide}
        self._last_calc = now
        return raw_tide

    async def _ensure_loaded(self) -> None:
        """Load Skyfield resources lazily (used for moon_phase and dawn/dusk period calculations)."""
        if self._sf_eph is not None and self._sf_ts is not None:
            return
        async with self._load_lock:
            if self._sf_eph is not None and self._sf_ts is not None:
                return

            def _blocking_load():
                sf_ts = self._loader.timescale()
                sf_eph = self._loader("de421.bsp")
                sf_wgs = wgs84
                try:
                    from skyfield import almanac as _almanac  # type: ignore
                except Exception:
                    _almanac = None
                version = getattr(skyfield, "__version__", "unknown")
                return sf_ts, sf_eph, sf_wgs, _almanac, version

            try:
                sf_ts, sf_eph, sf_wgs, sf_almanac, version = await self.hass.async_add_executor_job(_blocking_load)
                self._sf_ts = sf_ts
                self._sf_eph = sf_eph
                self._sf_wgs = sf_wgs
                self._sf_almanac = sf_almanac
                _LOGGER.info("Skyfield loaded version=%s", version)
            except Exception:
                _LOGGER.exception("Failed to load Skyfield resources")
                # Not fatal for tide heights, but will affect moon phase and dawn/dusk features.
                raise

    async def _async_find_next_moon_transit(self, sf_eph, sf_ts, sf_almanac, sf_wgs, start_dt: datetime) -> Optional[datetime]:
        # Preserve a small helper previously used by code that relied on Skyfield for moon transit
        try:
            start_dt = start_dt.astimezone(timezone.utc)
            t0 = sf_ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, start_dt.second)
            end_dt = start_dt + timedelta(days=3)
            t1 = sf_ts.utc(end_dt.year, end_dt.month, end_dt.day, end_dt.hour, end_dt.minute, end_dt.second)
            topos = sf_wgs.latlon(self.latitude, self.longitude)
            f = sf_almanac.meridian_transits(sf_eph, sf_eph["moon"], topos)
            times, events = sf_almanac.find_discrete(t0, t1, f)
            for t in times:
                try:
                    dt = t.utc_datetime().replace(tzinfo=timezone.utc)
                except Exception:
                    dt = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
                if dt and dt > start_dt:
                    return dt
            return None
        except Exception:
            _LOGGER.debug("Moon transit search failed", exc_info=True)
            return None

    async def compute_period_indices_for_timestamps(
        self,
        timestamps: Sequence[Any],
        mode: str = "full_day",
        dawn_window_hours: float = 1.0,
        *,
        location_tz: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute indices mapping for periods (same API as before).

        Note: this method still uses Skyfield for dawn/dusk when mode == 'dawn_dusk'.
        """
        if not location_tz:
            raise ValueError("location_tz is required (strict)")

        # parse incoming timestamps to UTC datetimes (aligned to canonical arrays)
        dt_objs: List[datetime] = []
        for ts in timestamps:
            try:
                parsed = dt_util.parse_datetime(str(ts))
                if parsed is None:
                    try:
                        v = float(ts)
                        if v > 1e12:
                            v = v / 1000.0
                        parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                    except Exception as exc:
                        raise ValueError(f"Unable to parse timestamp '{ts}': {exc}") from exc
            except Exception as exc:
                raise ValueError(f"Unable to parse timestamp '{ts}': {exc}") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)
        if not dt_objs:
            return {}

        # Build local-datetime list using ZoneInfo
        try:
            from zoneinfo import ZoneInfo
            tzinfo_local = ZoneInfo(location_tz)
        except Exception as exc:
            raise ValueError(f"Invalid location_tz '{location_tz}': {exc}") from exc

        local_dt_objs: List[datetime] = [dt.astimezone(tzinfo_local) for dt in dt_objs]
        index_dt_pairs = list(enumerate(dt_objs))  # dt_objs are UTC (for index matching)
        dates_needed = sorted({local_dt.date() for local_dt in local_dt_objs})

        # Load Skyfield only if dawn/dusk mode requested (for sunrise/sunset)
        if mode == "dawn_dusk":
            await self._ensure_loaded()
            sf_ts = self._sf_ts
            sf_eph = self._sf_eph
            sf_wgs = self._sf_wgs
            sf_almanac = self._sf_almanac
        else:
            sf_ts = sf_eph = sf_wgs = sf_almanac = None

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def _iso_z_local(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")

        for d in dates_needed:
            try:
                # local day start/end in local tz
                local_day_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tzinfo_local)
                local_day_end = local_day_start + timedelta(days=1)

                if mode == "dawn_dusk":
                    # Use Skyfield to compute sunrise/sunset as before
                    expand = timedelta(hours=12)
                    search_start_local = local_day_start - expand
                    search_end_local = local_day_end + expand

                    search_start_utc = search_start_local.astimezone(timezone.utc)
                    search_end_utc = search_end_local.astimezone(timezone.utc)
                    t0_exp = sf_ts.utc(search_start_utc.year, search_start_utc.month, search_start_utc.day, search_start_utc.hour, search_start_utc.minute, search_start_utc.second)
                    t1_exp = sf_ts.utc(search_end_utc.year, search_end_utc.month, search_end_utc.day, search_end_utc.hour, search_end_utc.minute, search_end_utc.second)

                    f = sf_almanac.sunrise_sunset(sf_eph, sf_wgs.latlon(self.latitude, self.longitude))
                    times, events = sf_almanac.find_discrete(t0_exp, t1_exp, f)
                    if not times:
                        raise RuntimeError(f"No sunrise/sunset events found for local date {d.isoformat()} at location lat={self.latitude},lon={self.longitude}")
                    sunrise_candidates: List[datetime] = []
                    sunset_candidates: List[datetime] = []
                    for t, ev in zip(times, events):
                        try:
                            evt_dt_utc = t.utc_datetime().replace(tzinfo=timezone.utc)
                        except Exception:
                            evt_dt_utc = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
                        evt_dt_local = evt_dt_utc.astimezone(tzinfo_local)
                        if bool(ev):
                            sunrise_candidates.append(evt_dt_local)
                        else:
                            sunset_candidates.append(evt_dt_local)
                    if not sunrise_candidates or not sunset_candidates:
                        raise RuntimeError(f"Unable to determine sunrise or sunset for local date {d.isoformat()} at lat={self.latitude},lon={self.longitude}")

                    morning_target = local_day_start + timedelta(hours=6)
                    evening_target = local_day_start + timedelta(hours=18)
                    sunrise_dt_local = min(sunrise_candidates, key=lambda e: abs((e - morning_target).total_seconds()))
                    sunset_dt_local = min(sunset_candidates, key=lambda e: abs((e - evening_target).total_seconds()))

                    dawn_start_local = sunrise_dt_local - timedelta(hours=dawn_window_hours)
                    dawn_end_local = sunrise_dt_local + timedelta(hours=dawn_window_hours)
                    dusk_start_local = sunset_dt_local - timedelta(hours=dawn_window_hours)
                    dusk_end_local = sunset_dt_local + timedelta(hours=dawn_window_hours)

                    dawn_start_utc = dawn_start_local.astimezone(timezone.utc)
                    dawn_end_utc = dawn_end_local.astimezone(timezone.utc)
                    dusk_start_utc = dusk_start_local.astimezone(timezone.utc)
                    dusk_end_utc = dusk_end_local.astimezone(timezone.utc)

                    date_key = d.isoformat()
                    result.setdefault(date_key, {})
                    dawn_indices: List[int] = []
                    dusk_indices: List[int] = []
                    for idx, dt in index_dt_pairs:
                        if dt >= dawn_start_utc and dt < dawn_end_utc:
                            dawn_indices.append(idx)
                        if dt >= dusk_start_utc and dt < dusk_end_utc:
                            dusk_indices.append(idx)
                    result[date_key]["dawn"] = {"indices": dawn_indices, "start": _iso_z_local(dawn_start_utc), "end": _iso_z_local(dawn_end_utc)}
                    result[date_key]["dusk"] = {"indices": dusk_indices, "start": _iso_z_local(dusk_start_utc), "end": _iso_z_local(dusk_end_utc)}
                else:
                    # full_day mode: define local periods 00-06,06-12,12-18,18-24 and convert to UTC for index matching
                    date_key = d.isoformat()
                    result.setdefault(date_key, {})
                    p00_start_local = local_day_start
                    p00_end_local = local_day_start + timedelta(hours=6)
                    p06_start_local = p00_end_local
                    p06_end_local = local_day_start + timedelta(hours=12)
                    p12_start_local = p06_end_local
                    p12_end_local = local_day_start + timedelta(hours=18)
                    p18_start_local = p12_end_local
                    p18_end_local = local_day_end
                    periods = [
                        ("period_00_06", p00_start_local, p00_end_local),
                        ("period_06_12", p06_start_local, p06_end_local),
                        ("period_12_18", p12_start_local, p12_end_local),
                        ("period_18_24", p18_start_local, p18_end_local),
                    ]
                    for pname, pstart_local, pend_local in periods:
                        pstart_utc = pstart_local.astimezone(timezone.utc)
                        pend_utc = pend_local.astimezone(timezone.utc)
                        indices: List[int] = []
                        for idx, dt in index_dt_pairs:
                            if dt >= pstart_utc and dt < pend_utc:
                                indices.append(idx)
                        result[date_key][pname] = {"indices": indices, "start": _iso_z_local(pstart_utc), "end": _iso_z_local(pend_utc)}
            except Exception as exc:
                _LOGGER.exception("compute_period_indices_for_timestamps failed for local date %s: %s", d.isoformat(), exc)
                raise
        return result