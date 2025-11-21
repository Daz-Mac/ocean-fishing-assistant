from __future__ import annotations
import logging
import math
import os
import json
import tempfile
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
from skyfield.framelib import ecliptic_frame  # use modern frames API instead of deprecated helpers
import skyfield  # for version reporting

import numpy as np

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# Constituent metadata: hours and canonical amplitude ratios (user-adjustable)
_CONSTITUENT_PERIOD_HOURS = {
    "M2": 12.4206,
    "S2": 12.0,
    "K1": 23.9345,
    "O1": 25.8193,
    "N2": 12.6583,
}
_CONSTITUENT_DEFAULT_RATIOS = {
    "M2": 1.0,
    "S2": 0.20,
    "K1": 0.25,
    "O1": 0.20,
    "N2": 0.15,
}

# Filename to optionally persist fitted coefficients (simple JSON)
_PERSIST_COEF_FILENAME = "tide_coeffs.json"


class TideProxy:
    """
    Astronomical tide proxy using Skyfield + small multi-constituent model.
    Skyfield data (ephemeris) will be stored in:
    <config_dir>/custom_components/ocean_fishing_assistant/data
    """

    def __init__(self, hass, latitude: float, longitude: float, ttl: int = _DEFAULT_TTL):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

        # persistence for fitted coefficients
        self._fitted_coef_vec: Optional[np.ndarray] = None

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

        # Try to load persisted coefficients if any
        self._load_persisted_coeffs()

    def _load_persisted_coeffs(self) -> None:
        try:
            path = os.path.join(self._data_dir, _PERSIST_COEF_FILENAME)
            if os.path.exists(path):
                with open(path, "r") as fh:
                    payload = json.load(fh)
                coef = payload.get("coef")
                if coef:
                    arr = np.asarray(coef, dtype=float)
                    self._fitted_coef_vec = arr
                    _LOGGER.info("Loaded persisted tide coefficients (%d values) from %s", arr.size, path)
        except Exception:
            _LOGGER.debug("Failed to load persisted coefficients", exc_info=True)

    def _persist_coeffs(self, coef_vec: np.ndarray) -> None:
        try:
            path = os.path.join(self._data_dir, _PERSIST_COEF_FILENAME)
            # atomic write via temporary file + replace
            dir_name = os.path.dirname(path)
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
                json.dump({"coef": coef_vec.tolist(), "ts": datetime.now(timezone.utc).isoformat()}, tmp)
                tmp_name = tmp.name
            os.replace(tmp_name, path)
            _LOGGER.info("Persisted tide coefficients (%d values) to %s", coef_vec.size, path)
        except Exception:
            _LOGGER.exception("Failed to persist tide coefficients to data dir")

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
                    version_tuple = getattr(skyfield, "VERSION", None)
                    return sf_ts, sf_eph, sf_wgs, sf_almanac, version, version_tuple
                except Exception:
                    _LOGGER.exception("Blocking Skyfield load failed")
                    raise

            try:
                sf_ts, sf_eph, sf_wgs, sf_almanac, version, version_tuple = await self.hass.async_add_executor_job(
                    _blocking_load
                )
                self._sf_ts = sf_ts
                self._sf_eph = sf_eph
                self._sf_wgs = sf_wgs
                self._sf_almanac = sf_almanac

                # Log and optionally warn about older Skyfield versions that may lack fixes.
                if version_tuple and isinstance(version_tuple, tuple):
                    try:
                        if version_tuple < (1, 48):
                            _LOGGER.warning(
                                "Ocean Fishing Assistant: Skyfield %s detected (tuple %s). "
                                "Consider upgrading to >=1.48 for improved accuracy and fixes.",
                                version,
                                version_tuple,
                            )
                    except Exception:
                        pass

                _LOGGER.info(
                    "Ocean Fishing Assistant: Skyfield %s loaded — using data directory: %s",
                    version,
                    self._data_dir,
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to initialize Skyfield Loader/ephemeris. Ensure 'skyfield' is installed and network access is available or pre-populate the data directory."
                )
                raise

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        """
        Compute tide data for the given ISO timestamps (expects UTC with trailing Z).
        Uses Skyfield for moon transits, phase and altitude. Returns a normalized dict.
        """
        now = dt_util.now().astimezone(timezone.utc)

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
            times_list = [sf_ts.from_datetime(dt) for dt in dt_objs]
        except Exception:
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
                astrom = (earth + topos).at(t).observe(moon_obj).apparent()
                alt, az, dist = astrom.altaz()
                deg = float(getattr(alt, "degrees", alt.degrees))
                moon_altitudes.append(deg)
        except Exception:
            _LOGGER.exception("Skyfield altitude calculation failed; returning None altitudes for timestamps")
            moon_altitudes = [None] * len(dt_objs)

        # Derive moon phase for each timestamp as a 0..1 synodic fraction using ecliptic longitudes
        moon_phases: List[Optional[float]] = []
        try:
            for t in times_list:
                sun_apparent = earth.at(t).observe(sun_obj).apparent()
                moon_apparent = earth.at(t).observe(moon_obj).apparent()
                sun_ecl = sun_apparent.frame_latlon(ecliptic_frame)
                moon_ecl = moon_apparent.frame_latlon(ecliptic_frame)
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                diff = (lon_moon - lon_sun) % 360.0
                moon_phases.append(diff / 360.0)
        except Exception:
            _LOGGER.exception("Skyfield phase calculation failed")
            raise

        # Choose representative scalar phase for tide strength/anchor computations.
        moon_phase_scalar: Optional[float] = None
        try:
            if moon_transit_dt is not None:
                t_anchor = sf_ts.from_datetime(moon_transit_dt)
                sun_app = earth.at(t_anchor).observe(sun_obj).apparent()
                moon_app = earth.at(t_anchor).observe(moon_obj).apparent()
                sun_ecl = sun_app.frame_latlon(ecliptic_frame)
                moon_ecl = moon_app.frame_latlon(ecliptic_frame)
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                moon_phase_scalar = (((lon_moon - lon_sun) % 360.0) / 360.0)
            else:
                moon_phase_scalar = moon_phases[0] if moon_phases else None
        except Exception:
            _LOGGER.exception("Failed to compute anchor moon phase")
            raise

        tide_strength = _compute_tide_strength(moon_phase_scalar)
        _LOGGER.debug("Moon phase scalar=%s tide_strength=%s", moon_phase_scalar, tide_strength)

        # Anchor: use precise moon_transit_dt if found, otherwise first timestamp or now
        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()

        base_amp = 1.0
        amp = base_amp * (0.5 + 0.5 * tide_strength)
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR

        # If we have a Skyfield-derived moon_transit_dt (topocentric), it already accounts for longitude.
        # Only apply a longitude adjustment when we are using a fallback anchor (e.g., first timestamp).
        if moon_transit_dt is None:
            lon_shift = (self.longitude / 360.0) * period_seconds
            t_anchor = anchor_epoch - lon_shift
            _LOGGER.debug("No moon_transit found — applying lon_shift=%s seconds to fallback anchor", lon_shift)
        else:
            t_anchor = anchor_epoch
            _LOGGER.debug("Using skyfield moon_transit_dt as anchor (no lon_shift applied) anchor_dt=%s", anchor_dt.isoformat().replace("+00:00", "Z"))

        # Build multi-constituent model (vectorized with numpy) anchored at t_anchor
        constituents = ["M2", "S2", "K1", "O1", "N2"]
        periods_sec = {k: (_CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in _CONSTITUENT_PERIOD_HOURS}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        # Default coefficients (a_k for cos, b_k for sin) using canonical ratios
        ratios = _CONSTITUENT_DEFAULT_RATIOS
        coef_list: List[float] = []
        for c in constituents:
            ratio = float(ratios.get(c, 0.0))
            A = amp * ratio
            coef_list.extend([A, 0.0])
        coef_vec = np.array(coef_list, dtype=float)  # [a_M2, b_M2, a_S2, b_S2, ...]

        # If persisted fitted coefficients exist and match shape, prefer those
        if self._fitted_coef_vec is not None and self._fitted_coef_vec.size == coef_vec.size:
            coef_vec = self._fitted_coef_vec.copy()
            _LOGGER.debug("Using persisted fitted coefficient vector")

        # Build design matrix X (n_samples x n_coeffs) where columns alternate cos(omega*t), sin(omega*t)
        t_rel = np.array([dt.timestamp() - t_anchor for dt in dt_objs], dtype=float)
        cols: List[np.ndarray] = []
        for c in constituents:
            w = omegas[c]
            cols.append(np.cos(w * t_rel))
            cols.append(np.sin(w * t_rel))
        if cols:
            X = np.column_stack(cols)
            y = X.dot(coef_vec)
        else:
            X = np.zeros((len(t_rel), 0), dtype=float)
            y = np.zeros((len(t_rel),), dtype=float)

        # produce tide_heights list from vectorized result
        tide_heights = [round(float(v), 3) for v in y.tolist()]

        # Scalar evaluators (use coef_vec and omegas for scalar evaluation and derivative)
        def _tide_val(epoch_sec: float) -> float:
            rel = epoch_sec - t_anchor
            v = 0.0
            for i, c in enumerate(constituents):
                w = omegas[c]
                a = float(coef_vec[2 * i])
                b = float(coef_vec[2 * i + 1])
                v += a * math.cos(w * rel) + b * math.sin(w * rel)
            return v

        def _tide_derivative(epoch_sec: float) -> float:
            rel = epoch_sec - t_anchor
            s = 0.0
            for i, c in enumerate(constituents):
                w = omegas[c]
                a = float(coef_vec[2 * i])
                b = float(coef_vec[2 * i + 1])
                s += (-a * w * math.sin(w * rel) + b * w * math.cos(w * rel))
            return s

        # Small bracketed root-finder to find a nearby extremum of the sum-of-sinusoids
        def _find_extremum_near(t_guess: float, is_max: bool = True, half_window: float = 10800.0) -> Tuple[float, float]:
            """
            Find extremum (max or min) near t_guess using derivative sign changes.
            half_window: seconds to search either side (default 3 hours)
            Returns (t_extremum, value_at_extremum)
            """
            a = t_guess - half_window
            b = t_guess + half_window
            # coarse scan to locate sign-change in derivative
            steps = 20
            xs = [a + i * (b - a) / steps for i in range(steps + 1)]
            vals = []
            for x in xs:
                try:
                    vals.append(_tide_derivative(x))
                except Exception:
                    vals.append(float("nan"))

            bracket = None
            for i in range(len(xs) - 1):
                fa = vals[i]
                fb = vals[i + 1]
                if math.isnan(fa) or math.isnan(fb):
                    continue
                if fa == 0.0:
                    bracket = (xs[i], xs[i])
                    break
                if fa * fb < 0.0:
                    bracket = (xs[i], xs[i + 1])
                    break

            if bracket is None:
                # fallback: return guess and its value
                return t_guess, _tide_val(t_guess)

            aa, bb = bracket
            fa = _tide_derivative(aa)
            fb = _tide_derivative(bb)

            # bisection refine
            tol = 1e-6
            max_iter = 60
            for _ in range(max_iter):
                m = 0.5 * (aa + bb)
                fm = _tide_derivative(m)
                if abs(fm) < tol:
                    val = _tide_val(m)
                    return m, val
                if fa * fm <= 0:
                    bb = m
                    fb = fm
                else:
                    aa = m
                    fa = fm
            m = 0.5 * (aa + bb)
            return m, _tide_val(m)

        # Analytic M2-based guess for next high/low as seeds; then refine using _find_extremum_near
        try:
            now_epoch = now.timestamp()
            quarter = period_seconds * 0.25
            # find smallest integer n such that t_anchor + quarter + n*period_seconds > now
            n = math.ceil((now_epoch - (t_anchor + quarter)) / period_seconds)
            analytic_next_high = t_anchor + quarter + n * period_seconds
            analytic_next_low = analytic_next_high + (period_seconds / 2.0)

            # refine using the multi-constituent model near the analytic guess
            next_high_epoch, next_high_val = _find_extremum_near(analytic_next_high, is_max=True, half_window=period_seconds / 4.0)
            next_low_epoch, next_low_val = _find_extremum_near(analytic_next_low, is_max=False, half_window=period_seconds / 4.0)

            next_high_dt = datetime.fromtimestamp(next_high_epoch, tz=timezone.utc)
            next_low_dt = datetime.fromtimestamp(next_low_epoch, tz=timezone.utc)
            next_high_height = round(float(next_high_val), 3)
            next_low_height = round(float(next_low_val), 3)
        except Exception:
            _LOGGER.exception("Error computing analytic/refined next high/low")
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

        # Attach small helper metadata for callers who want to calibrate:
        # NOTE: keep this small to avoid bloating entity attributes. Full persisted coeffs
        # are stored on disk via _persist_coeffs when calibrating.
        raw_tide["_helpers"] = {
            "constituents": constituents,
            "t_anchor": t_anchor,
            "period_seconds": period_seconds,
            "coef_vec_len": coef_vec.size,
            # include a callable-like representation note to explain the available helper _fit_coeffs_from_obs
            "calibration_info": "Call TideProxy.calibrate_from_observations(obs_timestamps, obs_heights, persist=True) to fit coefficients",
        }

        # Do NOT attempt to call DataFormatter.validate on this small tide-only dict.
        normalized = raw_tide
        self._cache = normalized
        self._last_calc = now
        return normalized

    def _fit_coeffs_from_obs(self, X: np.ndarray, obs_arr: np.ndarray, persist: bool = False) -> Optional[np.ndarray]:
        """
        Solve least-squares for coefficients given design matrix X and observation array obs_arr.
        If persist=True, persist results to disk and memory.
        Returns solution vector or None on failure.
        """
        try:
            if obs_arr.shape[0] != X.shape[0] or X.shape[1] == 0:
                _LOGGER.debug("Observation array shape mismatch or no design columns (obs_len=%d X.shape=%s)", obs_arr.shape[0], X.shape)
                return None
            # Solve least-squares
            sol, *_ = np.linalg.lstsq(X, obs_arr, rcond=None)
            if sol is None:
                return None
            if persist:
                try:
                    self._fitted_coef_vec = sol.copy()
                    self._persist_coeffs(self._fitted_coef_vec)
                except Exception:
                    _LOGGER.exception("Failed to persist fitted coefficients")
            return sol
        except Exception:
            _LOGGER.exception("Least-squares fit failed")
            return None

    async def calibrate_from_observations(
        self,
        obs_timestamps: Sequence[str],
        obs_heights: Sequence[float],
        persist: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Calibrate the multi-constituent coefficients from observed heights.
        - obs_timestamps: list of ISO timestamps (UTC) aligned with obs_heights
        - obs_heights: list of observed tide heights
        - persist: if True, save resulting coefficients to data directory for reuse

        Returns a dict with info (coef_vec, residual_norm, success) or None on failure.
        """
        try:
            if len(obs_timestamps) != len(obs_heights):
                raise ValueError("obs_timestamps and obs_heights must have same length")

            # Ensure model helpers are computed and Skyfield is loaded
            await self.get_tide_for_timestamps(obs_timestamps)
            helpers = self._cache.get("_helpers") if self._cache else None
            if not helpers:
                _LOGGER.debug("No helpers present to build design matrix; aborting calibration")
                return None
            constituents = helpers["constituents"]
            t_anchor = float(helpers["t_anchor"])
            coef_vec_len = int(helpers["coef_vec_len"])

            # build X
            times_epoch = np.array([dt_util.parse_datetime(str(ts)).replace(tzinfo=timezone.utc).timestamp() for ts in obs_timestamps], dtype=float)
            cols: List[np.ndarray] = []
            for c in constituents:
                if c not in _CONSTITUENT_PERIOD_HOURS:
                    raise ValueError(f"Unknown constituent {c}")
                period = _CONSTITUENT_PERIOD_HOURS[c] * _SECONDS_PER_HOUR
                w = 2.0 * math.pi / period
                rel = times_epoch - t_anchor
                cols.append(np.cos(w * rel))
                cols.append(np.sin(w * rel))

            if not cols:
                _LOGGER.error("No columns produced for calibration; aborting")
                return None

            X = np.column_stack(cols)
            obs_arr = np.asarray(obs_heights, dtype=float)

            # Guard: require at least (2 * num_constituents) samples for a stable fit
            num_const = len(constituents)
            min_samples = max(2 * num_const, 6)
            span_seconds = float(times_epoch.max() - times_epoch.min()) if len(times_epoch) > 1 else 0.0
            dominant_period = _CONSTITUENT_PERIOD_HOURS.get("M2", _TIDE_HALF_DAY_HOURS) * _SECONDS_PER_HOUR

            if len(obs_arr) < min_samples:
                raise ValueError(f"Not enough samples for calibration: got {len(obs_arr)}, need >= {min_samples}")
            if span_seconds < max(0.5 * dominant_period, dominant_period):
                _LOGGER.warning("Calibration data span is short (%.1f hours); results may be unstable", span_seconds / 3600.0)

            sol = self._fit_coeffs_from_obs(X, obs_arr, persist=persist)
            if sol is None:
                _LOGGER.error("Least-squares solver returned no solution")
                return None

            # compute residual norm
            residuals = X.dot(sol) - obs_arr
            residual_norm = float(np.linalg.norm(residuals))

            return {
                "coef_vec": sol.tolist(),
                "residual_norm": residual_norm,
                "success": True,
                "samples": len(obs_arr),
                "span_hours": span_seconds / 3600.0,
            }
        except Exception as exc:
            _LOGGER.exception("Calibration failed: %s", exc)
            return None

    async def _async_find_next_moon_transit(self, sf_eph, sf_ts, sf_almanac, sf_wgs, start_dt: datetime) -> Optional[datetime]:
        """
        Find the next moon meridian transit (local transit) after start_dt using skyfield.almanac.
        Returns timezone-aware UTC datetime or None on failure.
        """
        try:
            start_dt = start_dt.astimezone(timezone.utc)
            t0 = sf_ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, start_dt.second)
            end_dt = start_dt + timedelta(days=_ALMANAC_SEARCH_DAYS)
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
        """
        # Implementation unchanged from upstream; uses Skyfield and time parsing.
        dt_objs: List[datetime] = []
        for ts in timestamps:
            parsed = dt_util.parse_datetime(str(ts))
            if parsed is None:
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

        index_dt_pairs = list(enumerate(dt_objs))
        dates_needed = sorted({dt.date() for dt in dt_objs})

        await self._ensure_loaded()
        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac
        earth = sf_eph["earth"]
        topos = sf_wgs.latlon(self.latitude, self.longitude)
        sun = sf_eph["sun"]

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def _iso_z(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")

        for d in dates_needed:
            try:
                day_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)
                t0 = sf_ts.utc(day_start.year, day_start.month, day_start.day, 0, 0, 0)
                t1 = sf_ts.utc(day_end.year, day_end.month, day_end.day, 0, 0, 0)

                if mode == "dawn_dusk":
                    expand = timedelta(hours=12)
                    t0_exp = sf_ts.utc((day_start - expand).year, (day_start - expand).month, (day_start - expand).day, (day_start - expand).hour, (day_start - expand).minute, (day_start - expand).second)
                    t1_exp = sf_ts.utc((day_end + expand).year, (day_end + expand).month, (day_end + expand).day, (day_end + expand).hour, (day_end + expand).minute, (day_end + expand).second)
                    f = sf_almanac.sunrise_sunset(sf_eph, topos)
                    times, events = sf_almanac.find_discrete(t0_exp, t1_exp, f)
                    if not times:
                        raise RuntimeError(f"No sunrise/sunset events found for date {d.isoformat()} at location lat={self.latitude},lon={self.longitude}")
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
                    morning_target = day_start + timedelta(hours=6)
                    evening_target = day_start + timedelta(hours=18)
                    sunrise_dt = min(sunrise_candidates, key=lambda e: abs((e - morning_target).total_seconds()))
                    sunset_dt = min(sunset_candidates, key=lambda e: abs((e - evening_target).total_seconds()))
                    dawn_start = sunrise_dt - timedelta(hours=dawn_window_hours)
                    dawn_end = sunrise_dt + timedelta(hours=dawn_window_hours)
                    dusk_start = sunset_dt - timedelta(hours=dawn_window_hours)
                    dusk_end = sunset_dt + timedelta(hours=dawn_window_hours)
                    date_key = d.isoformat()
                    result.setdefault(date_key, {})
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
                    date_key = d.isoformat()
                    result.setdefault(date_key, {})
                    p00_start = day_start
                    p00_end = day_start + timedelta(hours=6)
                    p06_start = p00_end
                    p06_end = day_start + timedelta(hours=12)
                    p12_start = p06_end
                    p12_end = day_start + timedelta(hours=18)
                    p18_start = p12_end
                    p18_end = day_end
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
                raise
        return result


# -- module level helpers ----
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
        # circular distance to new moon: account for wrap-around at 1.0/0.0
        dist_new = min(abs(p - 0.0), abs(1.0 - p))
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        val = max(0.0, 1.0 - (dist / 0.25))
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5


def _predict_next_high_low(anchor_dt: datetime, now: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[datetime]]:
    try:
        if now is None:
            now = dt_util.now().astimezone(timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
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