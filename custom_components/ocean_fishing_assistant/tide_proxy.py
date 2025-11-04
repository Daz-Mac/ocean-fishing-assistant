from __future__ import annotations
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio

from homeassistant.util import dt as dt_util

# Local import to normalize output where available
try:
    from .data_formatter import DataFormatter
except Exception:
    DataFormatter = None  # optional

# Skyfield is required for astronomical tide calculations. Import at module load so missing
# dependency surfaces quickly during integration setup.
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield import almanac as _almanac  # type: ignore
import skyfield  # for version reporting

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield


class TideProxy:
    """
    Astronomical tide proxy using Skyfield only (no fallbacks).
    Skyfield data (ephemeris) will be stored in:
      <config_dir>/custom_components/ocean_fishing_assistant/data
    The Loader will download missing files into that directory on first use — but
    downloads and other blocking work are executed in the executor to avoid blocking
    Home Assistant's event loop.
    """

    def __init__(self, hass, latitude: float, longitude: float, ttl: int = _DEFAULT_TTL):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

        # prepare a dedicated data directory under the integration folder so Skyfield files
        # are consolidated and easy to pre-populate or inspect.
        try:
            data_dir = hass.config.path("custom_components", "ocean_fishing_assistant", "data")
        except Exception:
            # Fallback to joining paths if hass.config.path signature varies
            from homeassistant.const import CONFIG_DIR  # type: ignore

            data_dir = os.path.join(CONFIG_DIR, "custom_components", "ocean_fishing_assistant", "data")

        os.makedirs(data_dir, exist_ok=True)
        self._data_dir = data_dir

        # Create a Loader bound to the data directory. Do NOT call loader(...) or loader.timescale()
        # synchronously here because they may block on network/SSL operations.
        try:
            self._loader = Loader(self._data_dir)
        except Exception:
            # If Loader construction itself fails, log and re-raise so setup fails early.
            _LOGGER.exception("Failed to create Skyfield Loader. Ensure 'skyfield' is installed.")
            raise

        # Skyfield resources (populated lazily in executor)
        self._sf_ts = None
        self._sf_eph = None
        self._sf_wgs = None
        self._sf_almanac = None

        # lock to prevent concurrent background loads
        self._load_lock = asyncio.Lock()

        _LOGGER.debug("TideProxy initialized, data_dir=%s (Skyfield resources will load lazily)", self._data_dir)

    async def _ensure_loaded(self) -> None:
        """
        Ensure Skyfield resources are loaded. This method runs the actual blocking
        loader calls in an executor to avoid blocking the event loop.
        """
        # fast path
        if self._sf_eph is not None and self._sf_ts is not None:
            return

        async with self._load_lock:
            # double-check inside lock
            if self._sf_eph is not None and self._sf_ts is not None:
                return

            _LOGGER.debug("Loading Skyfield resources in executor (this may download ephemeris files)...")

            def _blocking_load():
                try:
                    sf_ts = self._loader.timescale()
                    sf_eph = self._loader("de421.bsp")
                    sf_wgs = wgs84
                    sf_almanac = _almanac
                    version = getattr(skyfield, "__version__", "unknown")
                    return sf_ts, sf_eph, sf_wgs, sf_almanac, version
                except Exception:
                    _LOGGER.exception("Blocking Skyfield load failed")
                    raise

            try:
                sf_ts, sf_eph, sf_wgs, sf_almanac, version = await self.hass.async_add_executor_job(_blocking_load)
                self._sf_ts = sf_ts
                self._sf_eph = sf_eph
                self._sf_wgs = sf_wgs
                self._sf_almanac = sf_almanac
                _LOGGER.info(
                    "Ocean Fishing Assistant: Skyfield %s loaded — using data directory: %s",
                    version,
                    self._data_dir,
                )
            except Exception:
                _LOGGER.exception("Failed to initialize Skyfield Loader/ephemeris. Ensure 'skyfield' is installed and network access is available or pre-populate the data directory.")
                raise

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        """
        Compute tide data for the given ISO timestamps (expects UTC with trailing Z).
        Uses Skyfield for moon transits, phase and altitude. Raises/logs on Skyfield runtime errors
        rather than silently falling back.
        """
        now = dt_util.now()

        fingerprint = (tuple(timestamps), round(self.latitude, 6), round(self.longitude, 6))
        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached = self._cache
            if cached.get("timestamps") == list(timestamps):
                return cached

        # Parse timestamps into TZ-aware UTC datetimes
        try:
            dt_objs = [datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc) for ts in timestamps]
        except Exception:
            empty = {
                "timestamps": list(timestamps),
                "tide_height_m": [None] * len(timestamps),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "none",
                "source": "astronomical_proxy",
            }
            return empty

        # Ensure skyfield resources are available (loads in executor if needed)
        try:
            await self._ensure_loaded()
        except Exception:
            # If loading failed, return a safe empty tide payload rather than crashing Home Assistant
            _LOGGER.exception("Skyfield resources not available; returning empty tide data")
            empty = {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "none",
                "source": "astronomical_unavailable",
            }
            return empty

        # Use Skyfield instance resources (initialized in _ensure_loaded)
        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        # Find precise moon transit after now (fail loudly on Skyfield errors)
        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            _LOGGER.exception("Skyfield transit-finding failed")
            raise

        # Compute moon altitudes for timestamps (used for state heuristics)
        moon_altitudes: List[Optional[float]] = []
        try:
            earth = sf_eph["earth"]
            moon = sf_eph["moon"]
            times_list = [sf_ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) for dt in dt_objs]
            loc = sf_wgs.latlon(self.latitude, self.longitude)
            for t in times_list:
                astrom = earth.at(t).observe(moon).apparent()
                alt, az, dist = astrom.altaz()
                moon_altitudes.append(float(getattr(alt, "degrees", alt.degrees)))
        except Exception:
            _LOGGER.exception("Skyfield altitude calculation failed")
            raise

        # Derive moon phase via elongation using Skyfield (0..1 where 0 new moon, 0.5 full)
        try:
            sample_dt = dt_objs[0] if dt_objs else now
            t0 = sf_ts.utc(sample_dt.year, sample_dt.month, sample_dt.day, sample_dt.hour, sample_dt.minute, sample_dt.second)
            earth = sf_eph["earth"]
            sun = sf_eph["sun"]
            moon = sf_eph["moon"]
            e = earth.at(t0)
            sun_astrom = e.observe(sun).apparent()
            moon_astrom = e.observe(moon).apparent()
            s = sun_astrom.position.km
            m = moon_astrom.position.km
            dot = s[0] * m[0] + s[1] * m[1] + s[2] * m[2]
            mag_s = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
            mag_m = math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
            cos_ang = max(-1.0, min(1.0, dot / (mag_s * mag_m)))
            angle = math.degrees(math.acos(cos_ang))  # 0..180
            moon_phase = (angle / 180.0) * 0.5
        except Exception:
            _LOGGER.exception("Skyfield phase calculation failed")
            raise

        tide_strength = _compute_tide_strength(moon_phase)

        # Anchor: use precise moon_transit_dt if found, otherwise first timestamp or now
        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp()

        base_amp = 1.0
        amp = base_amp * (0.5 + 0.5 * tide_strength)

        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR
        tide_heights: List[Optional[float]] = []
        for dt in dt_objs:
            sec = dt.timestamp()
            lon_shift = (self.longitude / 360.0) * period_seconds
            x = 2.0 * math.pi * ((sec - anchor_epoch + lon_shift) / period_seconds)
            value = amp * math.sin(x)
            tide_heights.append(round(float(value), 3))

        next_high_dt, next_low_dt = _predict_next_high_low(anchor_dt if anchor_dt else now)

        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phase,
            "tide_strength": float(round(tide_strength, 3)),
            "next_high": next_high_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_high_dt else "",
            "next_low": next_low_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_low_dt else "",
            "confidence": "astronomical",
            "source": "astronomical_skyfield",
        }

        # DataFormatter remains optional
        if DataFormatter:
            try:
                normalized = DataFormatter.format_tide_data(raw_tide)
            except Exception:
                _LOGGER.exception("DataFormatter failed to normalize tide data")
                raise
        else:
            normalized = raw_tide

        self._cache = normalized
        self._last_calc = now
        return normalized

    async def _async_find_next_moon_transit(self, sf_eph, sf_ts, sf_almanac, sf_wgs, start_dt: datetime) -> Optional[datetime]:
        """
        Find the next moon meridian transit (local transit) after start_dt using skyfield.almanac.
        Returns timezone-aware UTC datetime or raises on Skyfield errors.
        """
        # define search window
        t0 = sf_ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, start_dt.second)
        end_dt = start_dt + timedelta(days=_ALMANAC_SEARCH_DAYS)
        t1 = sf_ts.utc(end_dt.year, end_dt.month, end_dt.day, end_dt.hour, end_dt.minute, end_dt.second)

        # build observer/location
        loc = sf_wgs.latlon(self.latitude, self.longitude)

        # build function for meridian transits of the moon
        f = sf_almanac.meridian_transits(sf_eph, sf_eph["moon"], loc)

        # find discrete events
        times, events = sf_almanac.find_discrete(t0, t1, f)

        # choose first time that is strictly after start_dt
        for t in times:
            try:
                dt = t.utc_datetime().replace(tzinfo=timezone.utc)
            except Exception:
                dt = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
            if dt and dt > start_dt:
                return dt
        return None

    def _calculate_tide_state(self, moon_data: Dict[str, Optional[float]], sun_data: Dict[str, Optional[float]], now: datetime) -> str:
        try:
            moon_alt = moon_data.get("altitude")
            if moon_alt is None:
                rising = self._is_moon_rising(now)
                return "rising" if rising else "falling"
            if abs(moon_alt) > 70:
                return "slack_high"
            if abs(moon_alt) < 10:
                return "slack_low"
            return "rising" if self._is_moon_rising(now) else "falling"
        except Exception:
            _LOGGER.exception("Error calculating tide state")
            raise

    def _is_moon_rising(self, now: datetime) -> bool:
        try:
            lunar_day_hours = 24.84
            hours_since_epoch = now.timestamp() / 3600.0
            frac = (hours_since_epoch % lunar_day_hours) / lunar_day_hours
            return frac < 0.5
        except Exception:
            hour = now.hour + now.minute / 60.0
            return 0 < hour < 12.42


# -- module level helpers ----------------------------------------------------


def _coerce_phase(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            v = float(val)
            if v > 1.0:
                v = max(0.0, min(100.0, v)) / 100.0
            return max(0.0, min(1.0, v))
        s = str(val).strip().lower()
        names = {
            "new_moon": 0.0,
            "new": 0.0,
            "waxing_crescent": 0.125,
            "first_quarter": 0.25,
            "waxing_gibbous": 0.375,
            "full_moon": 0.5,
            "full": 0.5,
            "waning_gibbous": 0.625,
            "last_quarter": 0.75,
            "waning_crescent": 0.875,
            "0": 0.0,
            "0.0": 0.0,
        }
        if s in names:
            return float(names[s])
        try:
            f = float(s)
            if f > 1.0:
                f = max(0.0, min(100.0, f)) / 100.0
            return max(0.0, min(1.0, f))
        except Exception:
            return None
    except Exception:
        return None


def _compute_tide_strength(phase: Optional[float]) -> float:
    try:
        if phase is None:
            return 0.5
        p = float(phase)
        p = max(0.0, min(1.0, p))
        dist_new = abs(p - 0.0)
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        val = max(0.0, 1.0 - (dist / 0.25))
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5


def _predict_next_high_low(anchor_dt: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    try:
        now = dt_util.now()
        half_cycle = _TIDE_HALF_DAY_HOURS
        anchor_epoch = anchor_dt.timestamp()
        hours_since_anchor = (now.timestamp() - anchor_epoch) / 3600.0
        frac = (hours_since_anchor % half_cycle) / half_cycle
        remainder = (1.0 - frac) * half_cycle
        next_high_hours = remainder if remainder > 0.01 else half_cycle
        next_low_hours = next_high_hours + (half_cycle / 2.0)
        next_high = (now + timedelta(hours=next_high_hours)).astimezone(timezone.utc)
        next_low = (now + timedelta(hours=next_low_hours)).astimezone(timezone.utc)
        return next_high, next_low
    except Exception:
        _LOGGER.exception("Error predicting tide changes")
        raise