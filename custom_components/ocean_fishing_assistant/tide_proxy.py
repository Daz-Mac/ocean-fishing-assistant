from __future__ import annotations
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence
import pkgutil

_LOGGER = logging.getLogger(__name__)

# Semi-diurnal tide period (hours)
_TIDE_HALF_DAY_HOURS = 12.42

class TideProxy:
    """
    Astronomical tide proxy.

    Uses Skyfield (if available) to compute a lunar phase/strength and then
    generates a semi-diurnal sinusoidal tide prediction. This is intended as a
    defensive astronomical proxy (no external tide provider required).
    """

    def __init__(self, lat: float, lon: float, ttl: int = 15 * 60):
        self.latitude = float(lat)
        self.longitude = float(lon)
        self._ttl = int(ttl)
        self._cache: Optional[Dict[str, Any]] = None
        self._last_calc: Optional[datetime] = None

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        """
        Given an ordered sequence of ISO8601 UTC timestamps (strings),
        return a dict with keys:
          - timestamps: [iso...]
          - tide_height_m: [float...]  (aligned to input timestamps)
          - tide_phase: single phase value (0..1) or None
          - confidence, source
        """
        # produce numeric epoch seconds for interpolation
        try:
            dt_objs = [datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc) for ts in timestamps]
        except Exception:
            # fallback: return empty tide info
            return {"timestamps": list(timestamps), "tide_height_m": [None] * len(timestamps), "tide_phase": None, "confidence": "none", "source": "tide_proxy"}

        # compute moon phase strength using skyfield if available
        moon_phase = None
        strength = 0.5
        try:
            # lazy-import Skyfield
            from skyfield.api import load, N, wgs84  # type: ignore
            ts = load.timescale()
            eph = load('de421.bsp')  # Skyfield will download if necessary
            now = dt_objs[0] if dt_objs else datetime.now(timezone.utc)
            t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
            earth = eph['earth']
            sun = eph['sun']
            moon = eph['moon']
            # Phase approximation: angle between Sun-Earth-Moon
            e = earth.at(t)
            sun_astrom = e.observe(sun).apparent()
            moon_astrom = e.observe(moon).apparent()
            # elongation between sun and moon:
            # use position vectors to compute phase fraction
            from math import acos
            s = sun_astrom.position.km
            m = moon_astrom.position.km
            # vector angle:
            dot = s[0]*m[0] + s[1]*m[1] + s[2]*m[2]
            mag_s = math.sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
            mag_m = math.sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])
            cos_ang = max(-1.0, min(1.0, dot / (mag_s * mag_m)))
            angle = math.degrees(math.acos(cos_ang))  # 0..180
            # Map angle to phase fraction: 0 -> new (0.0), 180 -> full (0.5)
            moon_phase = (angle / 180.0) * 0.5
            # normalize to [0,1] mapping both new and full as springs:
            # choose p in [0,1] where 0=new,0.5=full,1->back to new
            # approximate: use cosine law mapping
            # For strength, distance to nearest spring (new or full)
            dist_new = abs(moon_phase - 0.0)
            dist_full = abs(moon_phase - 0.5)
            dist = min(dist_new, dist_full)
            strength = max(0.0, min(1.0, 1.0 - (dist / 0.25)))  # 1 at exact spring, 0 at quarter
        except Exception:
            _LOGGER.debug("Skyfield astro calculation not available or failed; using fallback", exc_info=True)
            moon_phase = None
            strength = 0.5

        # generate sinusoidal tide with amplitude derived from strength
        # base amplitude (meters) - conservative default
        base_amp = 1.0
        amp = base_amp * (0.5 + 0.5 * strength)  # 0.5..1.0 * base_amp -> 0.5..1.0m for neutral..spring
        period_seconds = _TIDE_HALF_DAY_HOURS * 3600.0
        # choose a reference epoch (use midnight UTC of first timestamp)
        if dt_objs:
            ref = datetime(dt_objs[0].year, dt_objs[0].month, dt_objs[0].day, tzinfo=timezone.utc)
        else:
            ref = datetime.now(timezone.utc)
        tide_heights: List[Optional[float]] = []
        for dt in dt_objs:
            # seconds since ref
            sec = (dt - ref).total_seconds()
            # phase shift: use longitude to shift local phase slightly (longitude/360 * period)
            lon_shift = (self.longitude / 360.0) * period_seconds
            x = 2.0 * math.pi * ((sec + lon_shift) / period_seconds)
            value = amp * math.sin(x)
            tide_heights.append(round(float(value), 3))
        return {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phase,
            "tide_strength": float(round(strength, 3)),
            "confidence": "proxy",
            "source": "astronomical_proxy",
        }