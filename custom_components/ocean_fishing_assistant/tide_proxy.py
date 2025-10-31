from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from homeassistant.util import dt as dt_util

# Local import to normalize output where available
try:
    from .data_formatter import DataFormatter
except Exception:
    DataFormatter = None  # optional

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0


class TideProxy:
    """
    Astronomical tide proxy using Skyfield and helpers.astro when available.

    How it works:
    - Try to reuse helpers.astro.calculate_astronomy_forecast(...) to get per-day
      moon_phase and moon_transit times (preferred).
    - Use Skyfield (lazy import) to compute instantaneous moon altitude for
      timestamps when available (useful to determine 'slack' states).
    - Generate a semi-diurnal sinusoidal tide aligned with moon_transit if present,
      otherwise use a longitude-derived phase shift. Amplitude scales with moon_phase
      (spring -> larger amplitude).
    - Return normalized structure with tide_height_m aligned to input timestamps,
      optional tide_phase, tide_strength, next_high/next_low prediction datetimes
      (ISO UTC strings), source and confidence.
    """

    def __init__(self, hass, latitude: float, longitude: float, ttl: int = _DEFAULT_TTL):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        """
        Accepts ISO8601 UTC timestamp strings (Z suffix).
        Returns normalized dict:
          {
            "timestamps": [...],
            "tide_height_m": [float|None,...],
            "tide_phase": float|None,  # 0..1
            "tide_strength": float,    # 0..1
            "next_high": "YYYY-MM-DDTHH:MM:SSZ" or "",
            "next_low": "...",
            "confidence": "proxy",
            "source": "astronomical_proxy",
            "forecast": { "YYYY-MM-DD": {...} }  # optional per-day summary
          }
        """
        now = dt_util.now()

        # simple cache by timestamps fingerprint (we include lat/lon implicitly)
        fingerprint = (tuple(timestamps), round(self.latitude, 6), round(self.longitude, 6))
        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            # cached response is valid (we don't key by fingerprint for simplicity),
            # but if user requests different timestamps we still regenerate below.
            cached = self._cache
            # Quick check: if timestamps match, return cached copy
            if cached.get("timestamps") == list(timestamps):
                return cached

        # Parse timestamps into TZ-aware UTC datetimes
        try:
            dt_objs = [datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc) for ts in timestamps]
        except Exception:
            # Malformed timestamps: return empty-shaped payload
            empty = {
                "timestamps": list(timestamps),
                "tide_height_m": [None] * len(timestamps),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "none",
                "source": "astronomical_proxy",
            }
            return empty

        # Obtain per-day astronomy forecast from helpers.astro when available
        astro_forecast: Dict[str, Any] = {}
        try:
            from .helpers.astro import calculate_astronomy_forecast  # type: ignore
            # request 7 days (today + future)
            astro_forecast = await calculate_astronomy_forecast(self.hass, self.latitude, self.longitude, days=7)
        except Exception:
            astro_forecast = {}

        # Determine moon_phase (0..1) preference using day's sample (prefer local moon_transit sample)
        moon_phase: Optional[float] = None
        moon_transit_dt: Optional[datetime] = None

        today_iso = dt_util.as_local(now).date().isoformat()
        if isinstance(astro_forecast, dict):
            today_entry = astro_forecast.get(today_iso) or {}
            if isinstance(today_entry, dict):
                mp = today_entry.get("moon_phase")
                if mp is not None:
                    try:
                        moon_phase = _coerce_phase(mp)
                    except Exception:
                        moon_phase = None
                # prefer moon_transit field if present (ISO string)
                mt = today_entry.get("moon_transit")
                if mt:
                    try:
                        moon_transit_dt = dt_util.parse_datetime(mt)
                        if moon_transit_dt and moon_transit_dt.tzinfo is None:
                            moon_transit_dt = moon_transit_dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        moon_transit_dt = None

        # If helpers.astro didn't provide moon_phase, try to compute via Skyfield (lazy)
        sf_ts = None
        sf_eph = None
        try:
            from skyfield.api import load, wgs84  # type: ignore
            sf_ts = load.timescale()
            # do not force download ephemeris here for the entire method; load() will obtain as needed
            sf_eph = load('de421.bsp')
        except Exception:
            sf_ts = None
            sf_eph = None

        # If we have skyfield, compute instantaneous moon altitude for the first timestamp (and optionally all)
        moon_altitudes: List[Optional[float]] = []
        if sf_ts and sf_eph:
            try:
                earth = sf_eph['earth']
                moon = sf_eph['moon']
                # Prepare times array for skyfield
                times_list = []
                for dt in dt_objs:
                    times_list.append(sf_ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
                # compute altaz for each time
                loc = wgs84.latlon(self.latitude, self.longitude)
                for t in times_list:
                    astrom = earth.at(t).observe(moon).apparent()
                    alt, az, dist = astrom.altaz()
                    # alt may be Angle-like; try degrees attribute
                    try:
                        moon_altitudes.append(float(getattr(alt, "degrees", alt.degrees)))
                    except Exception:
                        moon_altitudes.append(None)
            except Exception:
                moon_altitudes = [None] * len(dt_objs)
        else:
            moon_altitudes = [None] * len(dt_objs)

        # If moon_phase still unknown, derive an approximate phase via Sun-Moon elongation using Skyfield if available
        if moon_phase is None and sf_ts and sf_eph:
            try:
                # use first timestamp as sample
                sample_dt = dt_objs[0]
                t0 = sf_ts.utc(sample_dt.year, sample_dt.month, sample_dt.day, sample_dt.hour, sample_dt.minute, sample_dt.second)
                earth = sf_eph['earth']
                sun = sf_eph['sun']
                moon = sf_eph['moon']
                e = earth.at(t0)
                sun_astrom = e.observe(sun).apparent()
                moon_astrom = e.observe(moon).apparent()
                s = sun_astrom.position.km
                m = moon_astrom.position.km
                dot = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]
                mag_s = math.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
                mag_m = math.sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])
                cos_ang = max(-1.0, min(1.0, dot / (mag_s * mag_m)))
                angle = math.degrees(math.acos(cos_ang))  # 0..180
                # Map angle to phase fraction 0..0.5 (new..full) similar to earlier heuristics
                moon_phase = (angle / 180.0) * 0.5
            except Exception:
                moon_phase = None

        # compute tide_strength from moon_phase
        tide_strength = _compute_tide_strength(moon_phase)

        # Determine base phase anchor for sinusoid:
        # Prefer using moon_transit_dt if present (align high tide near transit), otherwise shift by longitude
        anchor_dt = moon_transit_dt or dt_objs[0] if dt_objs else now
        # convert anchor to epoch seconds
        anchor_epoch = anchor_dt.timestamp()

        # compute amplitude (meters) using tide_strength — conservative base amplitude
        base_amp = 1.0
        amp = base_amp * (0.5 + 0.5 * tide_strength)  # ranges ~0.5..1.0 * base_amp

        # compute tide for each requested timestamp, aligned to anchor; use semi-diurnal sinusoid
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR
        tide_heights: List[Optional[float]] = []
        for dt in dt_objs:
            sec = dt.timestamp()
            # local phase shift: use longitude to shift by fraction of day (longitude/360 * period)
            lon_shift = (self.longitude / 360.0) * period_seconds
            x = 2.0 * math.pi * ((sec - anchor_epoch + lon_shift) / period_seconds)
            value = amp * math.sin(x)
            tide_heights.append(round(float(value), 3))

        # Predict next high and low using semi-diurnal anchored to anchor_epoch (find next crest/trough)
        next_high_dt, next_low_dt = _predict_next_high_low(anchor_dt if anchor_dt else now)

        # Build optional forecast per-day using helpers.astro if available (use moon_transit/local noon sampling)
        forecast: Dict[str, Any] = {}
        try:
            # prefer using astro_forecast mapping if available
            if isinstance(astro_forecast, dict) and astro_forecast:
                for date_str, a in astro_forecast.items():
                    try:
                        # choose sample datetime: moon_transit if present else local noon UTC
                        dt_sample = None
                        if isinstance(a, dict):
                            mt = a.get("moon_transit")
                            if mt:
                                try:
                                    dt_sample = dt_util.parse_datetime(mt)
                                except Exception:
                                    dt_sample = None
                            if dt_sample is None:
                                # fallback to UTC noon
                                d = datetime.fromisoformat(date_str)
                                dt_sample = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=timezone.utc)
                        if dt_sample:
                            # compute day-level state & strength
                            phase_day = None
                            if isinstance(a, dict):
                                phase_day = a.get("moon_phase")
                            # try compute state via moon altitude at sample if skyfield present
                            alt = None
                            if sf_ts and sf_eph:
                                try:
                                    t = sf_ts.utc(dt_sample.year, dt_sample.month, dt_sample.day, dt_sample.hour, dt_sample.minute, dt_sample.second)
                                    earth = sf_eph['earth']
                                    moon = sf_eph['moon']
                                    loc = None
                                    from skyfield.api import wgs84  # type: ignore
                                    loc = wgs84.latlon(self.latitude, self.longitude)
                                    astrom = earth.at(t).observe(moon).apparent()
                                    alt_ang, az, dist = astrom.altaz()
                                    try:
                                        alt = float(getattr(alt_ang, "degrees", alt_ang.degrees))
                                    except Exception:
                                        alt = None
                                except Exception:
                                    alt = None
                            state_day = self._calculate_tide_state({"phase": phase_day, "altitude": alt}, {"elevation": None}, dt_sample)
                            strength_day = _compute_tide_strength(_coerce_phase(phase_day) if phase_day is not None else None)
                            forecast[date_str] = {
                                "state": state_day,
                                "strength": int(round(strength_day * 100)),
                                "datetime": dt_sample.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "source": "astronomical_calculation",
                            }
                    except Exception:
                        _LOGGER.debug("Skipping problematic astro forecast day %s", date_str, exc_info=True)
        except Exception:
            _LOGGER.debug("Per-day forecast generation failed", exc_info=True)

        raw_tide = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phase,
            "tide_strength": float(round(tide_strength, 3)),
            "next_high": next_high_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_high_dt else "",
            "next_low": next_low_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_low_dt else "",
            "confidence": "proxy",
            "source": "astronomical_proxy",
        }
        if forecast:
            raw_tide["forecast"] = forecast

        # Normalize via DataFormatter if available
        if DataFormatter:
            try:
                normalized = DataFormatter.format_tide_data(raw_tide)
            except Exception:
                normalized = raw_tide
        else:
            normalized = raw_tide

        # Cache and return
        self._cache = normalized
        self._last_calc = now
        return normalized

    def _calculate_tide_state(self, moon_data: Dict[str, Optional[float]], sun_data: Dict[str, Optional[float]], now: datetime) -> str:
        """
        Determine tide state (rising/falling/slack_high/slack_low/unknown).
        Accepts moon_data={"phase":..., "altitude":...}, sun_data={"elevation":...}.
        """
        try:
            moon_alt = moon_data.get("altitude")
            if moon_alt is None:
                # fallback to simple rising/falling heuristic
                rising = self._is_moon_rising(now)
                return "rising" if rising else "falling"
            # big positive altitude -> slack_high heuristic, very low -> slack_low
            if abs(moon_alt) > 70:
                return "slack_high"
            if abs(moon_alt) < 10:
                return "slack_low"
            return "rising" if self._is_moon_rising(now) else "falling"
        except Exception:
            _LOGGER.exception("Error calculating tide state")
            return "unknown"

    def _is_moon_rising(self, now: datetime) -> bool:
        """Simple heuristic: lunar cycle fraction to tell rising/falling."""
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
    """Convert numeric or named phase to float in [0,1]."""
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
    """
    Map moon phase to tide strength 0..1 (spring near new/full, neap near quarters).
    """
    try:
        if phase is None:
            return 0.5
        p = float(phase)
        p = max(0.0, min(1.0, p))
        dist_new = abs(p - 0.0)
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        # map [0..0.25] -> [1..0], clamp
        val = max(0.0, 1.0 - (dist / 0.25))
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5


def _predict_next_high_low(anchor_dt: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Using semi-diurnal half-cycle, estimate next high and next low relative to anchor datetime.
    Returns timezone-aware UTC datetimes.
    """
    try:
        now = dt_util.now()
        half_cycle = _TIDE_HALF_DAY_HOURS
        # figure fraction within half-cycle for now relative to anchor
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
        return None, None