# custom_components/ocean_fishing_assistant/tide_proxy.py
from __future__ import annotations
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncio

from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

# Skyfield (loaded lazily for moon/sun computations)
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield.framelib import ecliptic_frame
import skyfield

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_NEGATIVE_TTL_DEFAULT = 360  # seconds for negative-cache entries on failure

# Default World Tides base (configurable via const if needed)
from .const import (
    WORLD_TIDES_API_BASE,
    DOMAIN,
    COORD_ROUND_DECIMALS,
    TIDE_PROXY_TTL_DEFAULT,
)

# Minimal date formatting helper
def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


class TideProxy:
    """
    TideProxy backed by World Tides API using *only extremes* (high/low).
    Strict behavior: API key required at construction; if extremes are missing for requested window, raise.

    Caching / inflight dedupe:
      - Shared in-memory cache stored at hass.data[DOMAIN]["tide_api_cache"]
      - Inflight map stored at hass.data[DOMAIN]["tide_api_inflight"]
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
        store = self.hass.data.setdefault(DOMAIN, {})
        store.setdefault("tide_api_cache", {})
        store.setdefault("tide_api_inflight", {})
        return store

    def _rounded_coords(self) -> Tuple[float, float]:
        return (
            round(self.latitude, COORD_ROUND_DECIMALS),
            round(self.longitude, COORD_ROUND_DECIMALS),
        )

    def _time_bucket(self, dt: datetime, bucket_seconds: int = 300) -> int:
        return int(math.floor(dt.timestamp() / float(bucket_seconds)) * bucket_seconds)

    def _make_cache_key(self, endpoint: str, start_dt: datetime, end_dt: datetime, *, extra: Optional[Dict[str, Any]] = None) -> str:
        lat_r, lon_r = self._rounded_coords()
        start_b = self._time_bucket(start_dt, bucket_seconds=300)
        end_b = self._time_bucket(end_dt, bucket_seconds=300)
        parts = [endpoint, f"lat={lat_r}", f"lon={lon_r}", f"start={start_b}", f"end={end_b}"]
        if extra:
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
            try:
                del cache[key]
            except Exception:
                pass
            return None

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
        store = self._shared_store()
        cache = store.setdefault("tide_api_cache", {})
        ttl_use = int(ttl if ttl is not None else self._negative_ttl)
        expires_at = int(dt_util.now().timestamp()) + max(1, ttl_use)
        msg = str(exc)
        cache[key] = {"expires": expires_at, "error": True, "message": msg}
        _LOGGER.debug("Tide negative-cache set key=%s expires_in=%s message=%s", key, ttl_use, msg)

    async def _await_inflight_or_run(self, key: str, coro_func):
        store = self._shared_store()
        inflight = store.setdefault("tide_api_inflight", {})

        loop = asyncio.get_running_loop()
        existing = inflight.get(key)
        if existing:
            _LOGGER.debug("Awaiting inflight tide request key=%s", key)
            try:
                return await asyncio.shield(existing)
            finally:
                pass

        fut: asyncio.Future = loop.create_future()
        inflight[key] = fut
        try:
            _LOGGER.debug("Starting network request for tide key=%s", key)
            result = await coro_func()
            if not fut.done():
                fut.set_result(result)
            return result
        except Exception as exc:
            _LOGGER.debug("Tide request failed for key=%s exc=%s", key, exc)
            try:
                self._set_cached_error(key, exc, ttl=self._negative_ttl)
            except Exception:
                _LOGGER.debug("Failed to write negative cache for key=%s", key, exc_info=True)
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            try:
                inflight.pop(key, None)
            except Exception:
                pass

    # -----------------------
    # Fetch extremes only (cheap)
    # -----------------------
    async def _fetch_world_tides_extremes(self, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        """
        Request World Tides extremes (high/low) for [start_dt, end_dt].
        Uses shared cache and inflight de-duplication.
        Returns list of dicts: {'time': datetime(tz=UTC), 'height': float, 'type': 'high'|'low'|...}
        Raises RuntimeError on API problems or missing extremes.
        """
        key = self._make_cache_key("extremes", start_dt, end_dt)
        cached = None
        try:
            cached = self._get_cached(key)
        except RuntimeError:
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
            seconds = max(1, int((end_dt - start_dt).total_seconds()))
            length_days = max(1, math.ceil(seconds / 86400.0))

            session = async_get_clientsession(self.hass)

            params = {
                "lat": str(self.latitude),
                "lon": str(self.longitude),
                "start": str(int(start_dt.timestamp())),
                "days": str(int(length_days)),
                "key": self._api_key,
                # request extremes only (cheap)
                "extremes": "",
            }

            # redact key for logs
            safe_params = dict(params)
            if "key" in safe_params:
                safe_params["key"] = "REDACTED"

            url = f"{self._base}"
            _LOGGER.debug("WorldTides (v3) request: %s params=%s", url, safe_params)
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _LOGGER.error("WorldTides request failed status=%s body=%s", resp.status, body)
                        raise RuntimeError(f"WorldTides request failed status={resp.status}")
                    j = await resp.json()
            except Exception as exc:
                _LOGGER.exception("WorldTides request exception: %s", exc)
                raise RuntimeError(f"Failed to fetch World Tides data: {exc}") from exc

            # Attempt to find extremes in a few possible places in payload
            raw_extremes = None
            if "extremes" in j:
                raw_extremes = j.get("extremes")
            else:
                # some payloads may include a 'data' or 'predictions' list with type markers
                for k in ("data", "predictions"):
                    cand = j.get(k)
                    if not cand:
                        continue
                    # filter those entries that look like extremes (have 'type' or 'extreme' keys)
                    extremes_like = [it for it in cand if ("type" in it and ("high" in str(it.get("type")).lower() or "low" in str(it.get("type")).lower())) or ("extreme" in it)]
                    if extremes_like:
                        raw_extremes = extremes_like
                        break

            out_extremes_cacheable: List[Dict[str, Any]] = []
            out_extremes_parsed: List[Dict[str, Any]] = []
            if raw_extremes:
                for item in raw_extremes:
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
                    typ_s = (str(typ).strip() if typ is not None else "").lower()
                    # normalize type to "high" or "low" if possible
                    if typ_s.startswith("h"):
                        typ_norm = "high"
                    elif typ_s.startswith("l"):
                        typ_norm = "low"
                    else:
                        typ_norm = typ_s or None
                    out_extremes_parsed.append({"time": dt, "height": height_m, "type": typ_norm})
                    out_extremes_cacheable.append({"time": ts_out, "height": height_m, "type": typ_norm})

            if not out_extremes_parsed:
                _LOGGER.error("WorldTides response contained no usable extremes: %s", j)
                raise RuntimeError("WorldTides response contained no usable extremes")

            # sort & cache
            out_extremes_parsed.sort(key=lambda x: x["time"].timestamp())
            out_extremes_cacheable.sort(key=lambda x: x["time"])
            try:
                self._set_cached(key, out_extremes_cacheable, ttl=self._ttl)
            except Exception:
                _LOGGER.debug("Failed to set tide extremes cache for key=%s", key)
            return out_extremes_parsed

        # Use inflight/dedupe wrapper
        try:
            return await self._await_inflight_or_run(key, _do_fetch)
        except Exception:
            raise

    # -----------------------
    # High-level API (extremes-only)
    # -----------------------
    async def get_tide_for_timestamps(self, timestamps: Sequence[Any], *, location_tz: str) -> Dict[str, Any]:
        """
        Fetch tide predictions aligned to the provided timestamps using World Tides extremes only.

        Returns:
          - 'timestamps': list[ISOZ]
          - 'tide_phase': list[str] per timestamp ('rising'|'falling'|'flat')
          - 'tide_phase_name': list[str] friendly names
          - 'moon_phase': list[float or None] aligned numeric moon phase (0..1)
          - 'tide_strength': float 0..1 estimated from local extreme amplitudes
          - 'next_high'/'next_low': dict or None
        """
        if not location_tz:
            raise ValueError("location_tz is required (strict)")

        if not timestamps:
            return {
                "timestamps": [],
                "tide_phase": [],
                "moon_phase": [],
                "tide_strength": 0.0,
                "confidence": "worldtides_api",
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

        now = dt_util.now().astimezone(timezone.utc)
        try:
            new_keys = [_iso_z(dt) for dt in dt_objs]
        except Exception:
            new_keys = [int(dt.timestamp()) for dt in dt_objs]

        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached_keys = self._cache.get("timestamps")
            if cached_keys is not None and cached_keys == new_keys:
                return self._cache["raw_tide"]

        # Determine fetch window: expand by 1 day on each side to obtain bracketing extremes
        start_dt = min(dt_objs) - timedelta(days=1)
        end_dt = max(dt_objs) + timedelta(days=1)

        try:
            extremes_samples = await self._fetch_world_tides_extremes(start_dt, end_dt)
        except Exception as exc:
            _LOGGER.exception("WorldTides extremes fetch failed: %s", exc)
            raise RuntimeError(f"WorldTides extremes fetch failed: {exc}") from exc

        if not extremes_samples:
            raise RuntimeError("No extremes returned for requested window")

        # Build simple lists for binary search
        times_arr = [float(x["time"].timestamp()) for x in extremes_samples]
        heights_arr = [float(x["height"]) for x in extremes_samples]
        types_arr = [str(x.get("type") or "").lower() for x in extremes_samples]

        import bisect

        def _bracketing_indices(epoch_ts: float) -> Tuple[Optional[int], Optional[int]]:
            """
            Return (i_prev, i_next) where i_prev is index of last extreme with time <= epoch_ts,
            and i_next is index of first extreme with time > epoch_ts.
            Either may be None if not available.
            """
            i = bisect.bisect_right(times_arr, epoch_ts)
            i_prev = i - 1 if (i - 1) >= 0 else None
            i_next = i if i < len(times_arr) else None
            return i_prev, i_next

        tidal_heights_for_timestamps: List[float] = []
        tide_phase_list: List[str] = []
        tide_phase_name_list: List[str] = []

        # For each timestamp, compute interpolated height & phase using bracketing extremes
        for dt in dt_objs:
            epoch = float(dt.timestamp())
            i_prev, i_next = _bracketing_indices(epoch)
            # Strict: require both bracketing extremes to exist
            if i_prev is None or i_next is None:
                raise RuntimeError(f"Insufficient extremes to interpolate tide for timestamp {dt.isoformat()} (need surrounding high/low)")

            t_prev = float(times_arr[i_prev])
            t_next = float(times_arr[i_next])
            h_prev = float(heights_arr[i_prev])
            h_next = float(heights_arr[i_next])
            type_prev = types_arr[i_prev]
            type_next = types_arr[i_next]

            # Fraction of time between extremes
            if t_next == t_prev:
                frac = 0.0
            else:
                frac = (epoch - t_prev) / (t_next - t_prev)
                frac = max(0.0, min(1.0, frac))

            # Linear interpolation of height between extremes (approximation)
            interp_h = h_prev + (h_next - h_prev) * frac
            tidal_heights_for_timestamps.append(float(interp_h))

            # Determine phase: if prev is low and next is high -> rising; if prev high and next low -> falling
            if type_prev and type_next:
                if type_prev.startswith("l") and type_next.startswith("h"):
                    phase = "rising"
                    phase_name = "Rising"
                elif type_prev.startswith("h") and type_next.startswith("l"):
                    phase = "falling"
                    phase_name = "Falling"
                else:
                    # Unknown sequence: fallback to nearest trend by comparing heights
                    phase = "flat"
                    phase_name = "Flat"
            else:
                phase = "flat"
                phase_name = "Flat"
            tide_phase_list.append(phase)
            tide_phase_name_list.append(phase_name)

        # Determine next high/low from extremes relative to now
        now_ts = float(now.timestamp())
        next_high_obj = None
        next_low_obj = None
        for t, h, typ in zip(times_arr, heights_arr, types_arr):
            if t >= now_ts:
                if typ and typ.startswith("h") and next_high_obj is None:
                    next_high_obj = {"timestamp": datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")}
                if typ and typ.startswith("l") and next_low_obj is None:
                    next_low_obj = {"timestamp": datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")}
            if next_high_obj and next_low_obj:
                break

        # Estimate tide_strength as normalized amplitude (average high-low amplitude in window)
        try:
            # compute pairs of adjacent high/low amplitude differences
            amps: List[float] = []
            # scan for alternating high/low pairs
            for i in range(len(heights_arr) - 1):
                if types_arr[i] and types_arr[i + 1] and ((types_arr[i].startswith("h") and types_arr[i + 1].startswith("l")) or (types_arr[i].startswith("l") and types_arr[i + 1].startswith("h"))):
                    amps.append(abs(heights_arr[i] - heights_arr[i + 1]) / 2.0)
            if amps:
                amp = float(sum(amps) / len(amps))
            else:
                amp = max(heights_arr) - min(heights_arr)
            norm = max(0.0, min(1.0, amp / 2.5))  # same normalization as previous logic
            tide_strength_value = float(norm)
        except Exception:
            tide_strength_value = 0.5

        # Compute moon phases (try Skyfield; non-fatal)
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
            moon_phases = [None] * len(dt_objs)

        ts_isoz = [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs]

        raw_tide: Dict[str, Any] = {
            "timestamps": ts_isoz,
            "moon_phase": moon_phases,
            "tide_heights_m": tidal_heights_for_timestamps,
            "tide_phase": tide_phase_list,
            "tide_phase_name": tide_phase_name_list,
            "tide_strength": float(round(tide_strength_value, 3)),
            "confidence": "worldtides_api_extremes_only",
            "source": "worldtides_api",
            "_helpers": {
                "extremes_used": len(extremes_samples),
                "extremes_window_start": _iso_z(start_dt),
                "extremes_window_end": _iso_z(end_dt),
                "phase_offset_hours": float(self._phase_offset_hours),
            },
            "next_high": next_high_obj,
            "next_low": next_low_obj,
        }

        # Cache and return (per-instance cache)
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
                raise

    # compute_period_indices_for_timestamps preserved as before (uses Skyfield for dawn/dusk)
    async def compute_period_indices_for_timestamps(
        self,
        timestamps: Sequence[Any],
        mode: str = "full_day",
        dawn_window_hours: float = 1.0,
        *,
        location_tz: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        # (Implementation unchanged from previous version)
        if not location_tz:
            raise ValueError("location_tz is required (strict)")

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

        from zoneinfo import ZoneInfo
        tzinfo_local = ZoneInfo(location_tz)
        local_dt_objs: List[datetime] = [dt.astimezone(tzinfo_local) for dt in dt_objs]
        index_dt_pairs = list(enumerate(dt_objs))
        dates_needed = sorted({local_dt.date() for local_dt in local_dt_objs})

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
                local_day_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tzinfo_local)
                local_day_end = local_day_start + timedelta(days=1)

                if mode == "dawn_dusk":
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