from __future__ import annotations
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio

from homeassistant.util import dt as dt_util

# Local import left for historical reasons but not used to validate tide-only payloads.
try:
    from .data_formatter import DataFormatter  # type: ignore
except Exception:
    DataFormatter = None  # optional — but we will not call validate() on tide-only payloads

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
    The Loader will download missing files into that folder on first use — but
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
        Uses Skyfield for moon transits, phase and altitude. Returns a normalized dict.
        """
        now = dt_util.now()

        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached = self._cache
            if cached.get("timestamps") == list(timestamps):
                return cached

        # Parse timestamps into TZ-aware UTC datetimes
        try:
            dt_objs = []
            for ts in timestamps:
                try:
                    parsed = dt_util.parse_datetime(str(ts))
                    if parsed is None:
                        raise ValueError("parse_datetime returned None")
                    if parsed.tzinfo is None:
                        # assume UTC explicitly for naive timestamps
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    else:
                        parsed = parsed.astimezone(timezone.utc)
                    dt_objs.append(parsed)
                except Exception as exc:
                    raise ValueError(f"Unable to parse timestamp '{ts}': {exc}") from exc
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
            _LOGGER.exception("Skyfield resources not available; returning empty tide data")
            empty = {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "astronomical_unavailable",
                "source": "astronomical_unavailable",
            }
            return empty

        # Use Skyfield instance resources (initialized in _ensure_loaded)
        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        # Attempt to find moon transit; tolerate failure and continue with fallback anchor
        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            _LOGGER.debug("Failed to find next moon transit; continuing with fallback anchor", exc_info=True)
            moon_transit_dt = None

        # Prepare reusable Skyfield locals and Time objects for all timestamps
        try:
            times_list = [sf_ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) for dt in dt_objs]
        except Exception:
            # if we cannot build Time objects, fail loudly (strict)
            _LOGGER.exception("Failed to construct Skyfield Time objects for timestamps")
            raise

        try:
            earth = sf_eph["earth"]
            sun_obj = sf_eph["sun"]
            moon_obj = sf_eph["moon"]
            topos = sf_wgs.latlon(self.latitude, self.longitude)
        except Exception:
            _LOGGER.exception("Failed to resolve Skyfield bodies/topos")
            raise

        # Compute moon altitudes for timestamps (used for state heuristics)
        moon_altitudes: List[Optional[float]] = []
        try:
            for t in times_list:
                # Use earth + topos to get a Topos-relative observer with .at(t)
                astrom = (earth + topos).at(t).observe(moon_obj).apparent()
                alt, az, dist = astrom.altaz()
                # alt is an Angle; access degrees
                deg = float(getattr(alt, "degrees", alt.degrees))
                moon_altitudes.append(deg)
        except Exception:
            _LOGGER.exception("Skyfield altitude calculation failed; returning None altitudes for timestamps")
            moon_altitudes = [None] * len(dt_objs)

        # Derive moon phase for each timestamp as a 0..1 synodic fraction using ecliptic longitudes
        moon_phases: List[Optional[float]] = []
        try:
            for t in times_list:
                # compute apparent positions and convert to ecliptic longitudes
                sun_apparent = earth.at(t).observe(sun_obj).apparent()
                moon_apparent = earth.at(t).observe(moon_obj).apparent()
                sun_ecl = sun_apparent.ecliptic_latlon()
                moon_ecl = moon_apparent.ecliptic_latlon()
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                diff = (lon_moon - lon_sun) % 360.0
                moon_phases.append(diff / 360.0)
        except Exception:
            _LOGGER.exception("Skyfield phase calculation failed")
            # Strict behavior: surface Skyfield phase errors (do not silently continue)
            raise

        # Choose representative scalar phase for tide strength/anchor computations.
        # Prefer phase at moon_transit_dt if available, otherwise use first timestamp's phase.
        moon_phase_scalar: Optional[float] = None
        try:
            if moon_transit_dt is not None:
                # compute phase at the transit time separately (anchor)
                t_anchor = sf_ts.utc(moon_transit_dt.year, moon_transit_dt.month, moon_transit_dt.day, moon_transit_dt.hour, moon_transit_dt.minute, moon_transit_dt.second)
                sun_app = earth.at(t_anchor).observe(sun_obj).apparent()
                moon_app = earth.at(t_anchor).observe(moon_obj).apparent()
                sun_ecl = sun_app.ecliptic_latlon()
                moon_ecl = moon_app.ecliptic_latlon()
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                moon_phase_scalar = ( (lon_moon - lon_sun) % 360.0 ) / 360.0
            else:
                # fallback to first timestamp's computed phase
                moon_phase_scalar = moon_phases[0] if moon_phases else None
        except Exception:
            _LOGGER.exception("Failed to compute anchor moon phase")
            raise

        tide_strength = _compute_tide_strength(moon_phase_scalar)

        # Anchor: use precise moon_transit_dt if found, otherwise first timestamp or now
        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()

        base_amp = 1.0
        amp = base_amp * (0.5 + 0.5 * tide_strength)

        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR
        # phase shift from longitude (seconds)
        lon_shift = (self.longitude / 360.0) * period_seconds

        tide_heights: List[Optional[float]] = []
        for dt in dt_objs:
            sec = dt.timestamp()
            x = 2.0 * math.pi * ((sec - anchor_epoch + lon_shift) / period_seconds)
            try:
                value = amp * math.sin(x)
                tide_heights.append(round(float(value), 3))
            except Exception:
                tide_heights.append(None)

        # Analytic next high/low consistent with the sine model used above.
        try:
            now_epoch = now.timestamp()
            quarter = period_seconds * 0.25
            base = anchor_epoch - lon_shift + quarter
            n = math.ceil((now_epoch - base) / period_seconds)
            next_high_epoch = base + n * period_seconds
            next_low_epoch = next_high_epoch + (period_seconds / 2.0)
            next_high_dt = datetime.fromtimestamp(next_high_epoch, tz=timezone.utc)
            next_low_dt = datetime.fromtimestamp(next_low_epoch, tz=timezone.utc)
            # compute heights for analytic next high/low using same model
            xh = 2.0 * math.pi * ((next_high_epoch - anchor_epoch + lon_shift) / period_seconds)
            xl = 2.0 * math.pi * ((next_low_epoch - anchor_epoch + lon_shift) / period_seconds)
            next_high_height = round(float(amp * math.sin(xh)), 3)
            next_low_height = round(float(amp * math.sin(xl)), 3)
        except Exception:
            _LOGGER.exception("Error computing analytic next high/low")
            next_high_dt, next_low_dt = None, None
            next_high_height, next_low_height = None, None


        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            # per-timestamp tide heights (aligned)
            "tide_height_m": tide_heights,
            # per-timestamp tide_phase (aligned) — normalized 0..1 (new..full..new)
            "tide_phase": moon_phases,
            "tide_strength": float(round(tide_strength, 3)),
            "next_high": next_high_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_high_dt else "",
            "next_low": next_low_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_low_dt else "",
            "next_high_height_m": next_high_height,
            "next_low_height_m": next_low_height,
            "confidence": "astronomical" if moon_phase_scalar is not None else "astronomical_low_confidence",
            "source": "astronomical_skyfield",
        }

        # Do NOT attempt to call DataFormatter.validate on this small tide-only dict.
        # DataFormatter.validate is strict and expects an Open-Meteo shaped payload (hourly dict).
        normalized = raw_tide

        self._cache = normalized
        self._last_calc = now
        return normalized

    async def _async_find_next_moon_transit(self, sf_eph, sf_ts, sf_almanac, sf_wgs, start_dt: datetime) -> Optional[datetime]:
        """
        Find the next moon meridian transit (local transit) after start_dt using skyfield.almanac.
        Returns timezone-aware UTC datetime or None on failure.
        """
        try:
            # define search window
            t0 = sf_ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, start_dt.second)
            end_dt = start_dt + timedelta(days=_ALMANAC_SEARCH_DAYS)
            t1 = sf_ts.utc(end_dt.year, end_dt.month, end_dt.day, end_dt.hour, end_dt.minute, end_dt.second)

            # Topos for the location (used by almanac)
            topos = sf_wgs.latlon(self.latitude, self.longitude)

            # build function for meridian transits of the moon
            f = sf_almanac.meridian_transits(sf_eph, sf_eph["moon"], topos)

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
        except Exception:
            _LOGGER.debug("Moon transit search failed", exc_info=True)
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

    async def compute_period_indices_for_timestamps(
        self,
        timestamps: Sequence[str],
        mode: str = "full_day",
        dawn_window_hours: float = 1.0,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute period -> hourly-index mapping for the provided hourly timestamps.

        - timestamps: sequence of ISO timestamp strings or comparable values (expected UTC).
        - mode: "dawn_dusk" or "full_day" (4 periods).
        - dawn_window_hours: half-width of dawn/dusk window in hours (±dawn_window_hours around sunrise/sunset).

        Returns mapping:
          { "YYYY-MM-DD": { "period_name": { "indices": [idx,...], "start": "ISOZ", "end": "ISOZ" }, ... }, ... }

        Strict behavior:
          - If Skyfield cannot compute sunrise/sunset for a required date, raise an exception.
          - All datetimes in returned mapping are timezone-aware UTC ISO strings ending with Z.
        """
        # Parse timestamps into aware UTC datetimes
        dt_objs: List[datetime] = []
        for ts in timestamps:
            parsed = dt_util.parse_datetime(str(ts))
            if parsed is None:
                # last-resort numeric epoch
                try:
                    v = float(ts)
                    if v > 1e12:
                        v = v / 1000.0
                    parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception as exc:
                    raise ValueError(f"Unable to parse timestamp '{ts}': {exc}") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)

        if not dt_objs:
            return {}

        # Build mapping of index -> dt for quick reference
        index_dt_pairs = list(enumerate(dt_objs))

        # Determine set of dates we need to cover (based on timestamp datetimes)
        dates_needed = sorted({dt.date() for dt in dt_objs})

        # Ensure skyfield loaded
        await self._ensure_loaded()
        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        earth = sf_eph["earth"]
        topos = sf_wgs.latlon(self.latitude, self.longitude)
        sun = sf_eph["sun"]

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Helper to produce ISOZ string from aware UTC datetime
        def _iso_z(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")

        # Pre-calc seconds for small offset checks
        for d in dates_needed:
            try:
                # search window: from 00:00 UTC of date d to 00:00 UTC of next day
                day_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)

                t0 = sf_ts.utc(day_start.year, day_start.month, day_start.day, 0, 0, 0)
                t1 = sf_ts.utc(day_end.year, day_end.month, day_end.day, 0, 0, 0)

                # For sunrise/sunset mode, use almanac.sunrise_sunset
                if mode == "dawn_dusk":
                    # Expand search window to +/- 12 hours to catch events that fall outside strict UTC date
                    # (helps when local sunrise/sunset fall on adjacent UTC dates).
                    expand = timedelta(hours=12)
                    t0_exp = sf_ts.utc((day_start - expand).year, (day_start - expand).month, (day_start - expand).day, (day_start - expand).hour, (day_start - expand).minute, (day_start - expand).second)
                    t1_exp = sf_ts.utc((day_end + expand).year, (day_end + expand).month, (day_end + expand).day, (day_end + expand).hour, (day_end + expand).minute, (day_end + expand).second)

                    f = sf_almanac.sunrise_sunset(sf_eph, topos)
                    times, events = sf_almanac.find_discrete(t0_exp, t1_exp, f)

                    # If find_discrete returns nothing (polar day/night or other issues), raise strict error
                    if not times:
                        raise RuntimeError(f"No sunrise/sunset events found for date {d.isoformat()} at location lat={self.latitude},lon={self.longitude}")

                    # Use the events array directly to classify sunrise/sunset:
                    # - events[i] is truthy when the Sun is above horizon after times[i] => sunrise candidate
                    # - events[i] is falsy when the Sun is below horizon after times[i] => sunset candidate
                    sunrise_candidates: List[datetime] = []
                    sunset_candidates: List[datetime] = []
                    evt_dt_list: List[datetime] = []

                    for t, ev in zip(times, events):
                        try:
                            evt_dt = t.utc_datetime().replace(tzinfo=timezone.utc)
                        except Exception:
                            evt_dt = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
                        evt_dt_list.append(evt_dt)
                        if bool(ev):
                            sunrise_candidates.append(evt_dt)
                        else:
                            sunset_candidates.append(evt_dt)

                    _LOGGER.debug(
                        "Skyfield sunrise/sunset discrete events for date=%s lat=%s lon=%s -> times=%s events=%s",
                        d.isoformat(),
                        self.latitude,
                        self.longitude,
                        [e.isoformat().replace("+00:00", "Z") for e in evt_dt_list],
                        list(map(int, list(events))),
                    )

                    # Strict: if we failed to classify either sunrise or sunset, raise (no fallback)
                    if not sunrise_candidates or not sunset_candidates:
                        _LOGGER.error(
                            "Failed to classify sunrise/sunset by event flags for date %s at lat=%s,lon=%s; sunrise_candidates=%s sunset_candidates=%s",
                            d.isoformat(),
                            self.latitude,
                            self.longitude,
                            [s.isoformat().replace("+00:00", "Z") for s in sunrise_candidates],
                            [s.isoformat().replace("+00:00", "Z") for s in sunset_candidates],
                        )
                        raise RuntimeError(f"Unable to determine sunrise or sunset for date {d.isoformat()} at lat={self.latitude},lon={self.longitude}")

                    # choose the most relevant sunrise/sunset for the local date:
                    morning_target = day_start + timedelta(hours=6)
                    evening_target = day_start + timedelta(hours=18)
                    sunrise_dt = min(sunrise_candidates, key=lambda e: abs((e - morning_target).total_seconds()))
                    sunset_dt = min(sunset_candidates, key=lambda e: abs((e - evening_target).total_seconds()))

                    # Build dawn/dusk windows (±dawn_window_hours)
                    dawn_start = sunrise_dt - timedelta(hours=dawn_window_hours)
                    dawn_end = sunrise_dt + timedelta(hours=dawn_window_hours)
                    dusk_start = sunset_dt - timedelta(hours=dawn_window_hours)
                    dusk_end = sunset_dt + timedelta(hours=dawn_window_hours)

                    date_key = d.isoformat()
                    result.setdefault(date_key, {})

                    # iterate hourly timestamps and include indices that fall within the windows (absolute datetime compare)
                    dawn_indices: List[int] = []
                    dusk_indices: List[int] = []
                    for idx, dt in index_dt_pairs:
                        if dt >= dawn_start and dt < dawn_end:
                            dawn_indices.append(idx)
                        if dt >= dusk_start and dt < dusk_end:
                            dusk_indices.append(idx)

                    result[date_key]["dawn"] = {"indices": dawn_indices, "start": _iso_z(dawn_start), "end": _iso_z(dawn_end)}
                    result[date_key]["dusk"] = {"indices": dusk_indices, "start": _iso_z(dusk_start), "end": _iso_z(dusk_end)}

                else:
                    # Full day mode: build 4 absolute periods per date using absolute datetimes
                    date_key = d.isoformat()
                    result.setdefault(date_key, {})

                    # period boundaries (absolute)
                    p00_start = day_start
                    p00_end = day_start + timedelta(hours=6)
                    p06_start = p00_end
                    p06_end = day_start + timedelta(hours=12)
                    p12_start = p06_end
                    p12_end = day_start + timedelta(hours=18)
                    p18_start = p12_end
                    p18_end = day_end  # 24:00 of that date

                    periods = [
                        ("period_00_06", p00_start, p00_end),
                        ("period_06_12", p06_start, p06_end),
                        ("period_12_18", p12_start, p12_end),
                        ("period_18_24", p18_start, p18_end),
                    ]

                    for pname, pstart, pend in periods:
                        indices: List[int] = []
                        for idx, dt in index_dt_pairs:
                            if dt >= pstart and dt < pend:
                                indices.append(idx)
                        result[date_key][pname] = {"indices": indices, "start": _iso_z(pstart), "end": _iso_z(pend)}

            except Exception as exc:
                _LOGGER.exception("compute_period_indices_for_timestamps failed for date %s: %s", d.isoformat(), exc)
                # Strict: propagate error so caller can fail loudly
                raise

        return result


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