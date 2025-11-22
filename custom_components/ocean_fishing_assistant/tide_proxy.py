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

# Optional local formatter (kept optional)
try:
    from .data_formatter import DataFormatter  # type: ignore
except Exception:
    DataFormatter = None

# Skyfield is required for astronomical helpers (loaded lazily)
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield import almanac as _almanac  # type: ignore
from skyfield.framelib import ecliptic_frame
import skyfield  # for version reporting

import numpy as np  # numpy is used for vectorized evaluation and fitting

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# Constituent metadata
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

_PERSIST_COEF_FILENAME = "tide_coeffs.json"


class TideProxy:
    """
    TideProxy: uses Skyfield for astronomical helpers and a strict persisted
    multi-constituent harmonic model for tide heights.

    Important policy: tide heights are computed ONLY from persisted coefficients.
    If coefficients are missing or invalid, get_tide_for_timestamps will return a
    failure payload with confidence 'coefficients_missing'. Helpers (t_anchor,
    constituents, omegas) are still computed and returned to assist calibration.
    """

    def __init__(self, hass, latitude: float, longitude: float, ttl: int = _DEFAULT_TTL):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

        # persisted fitted coefficient vector (numpy 1-D array, 2 * num_constituents)
        self._fitted_coef_vec: Optional[np.ndarray] = None

        # keep last helpers produced (constituents, t_anchor, period_seconds, coef_vec_len)
        self._last_helpers: Optional[Dict[str, Any]] = None

        # prepare a dedicated data directory under the integration folder
        try:
            data_dir = hass.config.path("custom_components", "ocean_fishing_assistant", "data")
        except Exception:
            from homeassistant.const import CONFIG_DIR  # type: ignore
            data_dir = os.path.join(CONFIG_DIR, "custom_components", "ocean_fishing_assistant", "data")

        os.makedirs(data_dir, exist_ok=True)
        self._data_dir = data_dir

        # Create Skyfield Loader bound to that directory
        try:
            self._loader = Loader(self._data_dir)
        except Exception:
            _LOGGER.exception("Failed to create Skyfield Loader. Ensure 'skyfield' is installed.")
            raise

        # Skyfield resources (populated lazily)
        self._sf_ts = None
        self._sf_eph = None
        self._sf_wgs = None
        self._sf_almanac = None

        # lock to prevent concurrent loads
        self._load_lock = asyncio.Lock()

        _LOGGER.debug("TideProxy initialized, data_dir=%s (Skyfield resources will load lazily)", self._data_dir)

        # NOTE: do NOT load persisted coeffs synchronously here (would block event loop).
        # Use async_load_persisted_coeffs() from async setup to trigger a safe load.

    # ------------------ Persistence helpers ------------------

    def _load_persisted_coeffs(self) -> None:
        """
        Blocking (synchronous) load of persisted coefficients. Must be called via
        async_load_persisted_coeffs() to avoid blocking HA event loop.
        """
        try:
            path = os.path.join(self._data_dir, _PERSIST_COEF_FILENAME)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                coef = payload.get("coef")
                if coef:
                    arr = np.asarray(coef, dtype=float)
                    self._fitted_coef_vec = arr
                    # also attempt to load saved constituents and t_anchor if present
                    helpers = {}
                    if "constituents" in payload:
                        helpers["constituents"] = payload.get("constituents")
                    if "t_anchor" in payload:
                        try:
                            helpers["t_anchor"] = float(payload.get("t_anchor"))
                        except Exception:
                            helpers["t_anchor"] = None
                    if helpers:
                        self._last_helpers = helpers
                    _LOGGER.info("Loaded persisted tide coefficients (%d values) from %s", arr.size, path)
        except Exception:
            _LOGGER.debug("Failed to load persisted coefficients", exc_info=True)

    async def async_load_persisted_coeffs(self) -> None:
        """
        Async wrapper to load persisted coefficients without blocking the event loop.
        Home Assistant (async_setup_entry) calls this — it runs the blocking loader
        in the executor so open()/json.load() do not block.
        """
        try:
            await self.hass.async_add_executor_job(self._load_persisted_coeffs)
        except Exception:
            _LOGGER.exception("async_load_persisted_coeffs failed")

    def _persist_coeffs(self, coef_vec: np.ndarray, constituents: Optional[List[str]] = None, t_anchor: Optional[float] = None) -> None:
        try:
            path = os.path.join(self._data_dir, _PERSIST_COEF_FILENAME)
            dir_name = os.path.dirname(path)
            # Build payload to persist; include helpers when available
            payload = {"coef": coef_vec.tolist(), "ts": datetime.now(timezone.utc).isoformat()}
            if constituents:
                payload["constituents"] = constituents
            elif self._last_helpers and "constituents" in self._last_helpers:
                payload["constituents"] = self._last_helpers["constituents"]
            if t_anchor is not None:
                payload["t_anchor"] = float(t_anchor)
            elif self._last_helpers and "t_anchor" in self._last_helpers:
                payload["t_anchor"] = float(self._last_helpers["t_anchor"])
            # atomic write
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
                json.dump(payload, tmp)
                tmp_name = tmp.name
            os.replace(tmp_name, path)
            _LOGGER.info("Persisted tide coefficients (%d values) to %s", coef_vec.size, path)
        except Exception:
            _LOGGER.exception("Failed to persist tide coefficients to data dir")

    # ------------------ Skyfield lazy load ------------------

    async def _ensure_loaded(self) -> None:
        """
        Ensure Skyfield resources are loaded; runs the loader in executor to avoid blocking.
        """
        if self._sf_eph is not None and self._sf_ts is not None:
            return

        async with self._load_lock:
            if self._sf_eph is not None and self._sf_ts is not None:
                return

            _LOGGER.debug("Loading Skyfield resources in executor (may download ephemeris files)...")

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

                if version_tuple and isinstance(version_tuple, tuple):
                    try:
                        if version_tuple < (1, 48):
                            _LOGGER.warning(
                                "Ocean Fishing Assistant: Skyfield %s detected (tuple %s). Consider upgrading to >=1.48.",
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
                    "Failed to initialize Skyfield Loader/ephemeris. Ensure 'skyfield' is installed and data dir is accessible."
                )
                raise

    # ------------------ Tide computation (strict persisted coefficients) ------------------

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        """
        Compute tide payload for given ISO timestamps (UTC expected).
        IMPORTANT: tide heights are computed only when persisted coefficients are present and valid.
        Otherwise returns a failure payload with 'confidence' set to 'coefficients_missing'.
        """
        now = dt_util.now().astimezone(timezone.utc)

        # cache fast path
        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached = self._cache
            if cached.get("timestamps") == list(timestamps):
                return cached

        # Parse timestamps into tz-aware UTC datetimes
        try:
            dt_objs: List[datetime] = []
            for ts in timestamps:
                parsed = dt_util.parse_datetime(str(ts))
                if parsed is None:
                    raise ValueError(f"parse_datetime returned None for {ts}")
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                else:
                    parsed = parsed.astimezone(timezone.utc)
                dt_objs.append(parsed)
        except Exception as exc:
            _LOGGER.exception("Failed to parse provided timestamps: %s", exc)
            empty = {
                "timestamps": list(timestamps),
                "tide_height_m": [None] * len(timestamps),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "bad_timestamps",
                "source": "tide_proxy",
            }
            return empty

        # Ensure Skyfield available for helpers
        try:
            await self._ensure_loaded()
        except Exception:
            _LOGGER.exception("Skyfield unavailable; returning failure payload")
            empty = {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "astronomical_unavailable",
                "source": "astronomical_unavailable",
            }
            self._cache = empty
            self._last_calc = now
            return empty

        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        # Attempt to find next moon transit (optional) to use as t_anchor
        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            _LOGGER.debug("Failed to find next moon transit; will fallback to timestamp anchor", exc_info=True)
            moon_transit_dt = None

        # times_list for skyfield ops
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

        # compute moon altitudes (best-effort)
        moon_altitudes: List[Optional[float]] = []
        try:
            for t in times_list:
                astrom = (earth + topos).at(t).observe(moon_obj).apparent()
                alt, az, dist = astrom.altaz()
                deg = float(getattr(alt, "degrees", getattr(alt, "degrees", None)))
                moon_altitudes.append(deg)
        except Exception:
            _LOGGER.exception("Skyfield altitude calculation failed; falling back to None altitudes")
            moon_altitudes = [None] * len(dt_objs)

        # compute moon phases (0..1) per timestamp
        moon_phases: List[Optional[float]] = []
        try:
            for t in times_list:
                sun_app = earth.at(t).observe(sun_obj).apparent()
                moon_app = earth.at(t).observe(moon_obj).apparent()
                sun_ecl = sun_app.frame_latlon(ecliptic_frame)
                moon_ecl = moon_app.frame_latlon(ecliptic_frame)
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                diff = (lon_moon - lon_sun) % 360.0
                moon_phases.append(diff / 360.0)
        except Exception:
            _LOGGER.exception("Skyfield phase calculation failed")
            raise

        # representative phase scalar (anchor)
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

        # Anchor: prefer precise moon_transit_dt; otherwise use first timestamp
        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()

        base_amp = 1.0
        amp = base_amp * (0.5 + 0.5 * tide_strength)
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR

        # apply simple lon shift if using fallback anchor (not required when moon_transit_dt present)
        if moon_transit_dt is None:
            lon_shift = (self.longitude / 360.0) * period_seconds
            t_anchor = anchor_epoch - lon_shift
            _LOGGER.debug("No moon_transit found — applying lon_shift=%s seconds to fallback anchor", lon_shift)
        else:
            t_anchor = anchor_epoch
            _LOGGER.debug("Using skyfield moon_transit_dt as anchor (no lon_shift) anchor_dt=%s", anchor_dt.isoformat().replace("+00:00", "Z"))

        # Build model helpers (constituents, omegas)
        constituents = ["M2", "S2", "K1", "O1", "N2"]
        periods_sec = {k: (_CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in _CONSTITUENT_PERIOD_HOURS}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        # Save helpers for calibration usage
        helpers = {
            "constituents": constituents,
            "t_anchor": float(t_anchor),
            "period_seconds": float(period_seconds),
            "coef_vec_len": 2 * len(constituents),
        }
        self._last_helpers = helpers

        # Strict policy: require persisted coefficients to compute tide heights
        if self._fitted_coef_vec is None:
            _LOGGER.error("Persisted tide coefficients required but missing. Aborting evaluation (no fallback).")
            normalized = {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": moon_phases,
                "tide_strength": float(round(tide_strength, 3)),
                "next_high": "",
                "next_low": "",
                "next_high_height_m": None,
                "next_low_height_m": None,
                "confidence": "coefficients_missing",
                "source": "coefficients_missing",
                "_helpers": helpers,
            }
            self._cache = normalized
            self._last_calc = now
            return normalized

        # Validate persisted coefficients shape
        coef_arr = np.asarray(self._fitted_coef_vec, dtype=float)
        n_pairs = coef_arr.size // 2
        if coef_arr.size % 2 != 0 or n_pairs != len(constituents):
            _LOGGER.error(
                "Persisted coefficient vector shape mismatch: coef_len=%d expected_pairs=%d constituents=%s",
                coef_arr.size,
                len(constituents),
                constituents,
            )
            normalized = {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": moon_phases,
                "tide_strength": float(round(tide_strength, 3)),
                "next_high": "",
                "next_low": "",
                "next_high_height_m": None,
                "next_low_height_m": None,
                "confidence": "coefficients_invalid",
                "source": "coefficients_invalid",
                "_helpers": helpers,
            }
            self._cache = normalized
            self._last_calc = now
            return normalized

        # Evaluate persisted coefficients using numpy (vectorized)
        try:
            A = coef_arr[0::2].astype(float)
            B = coef_arr[1::2].astype(float)
            t_rel = np.array([dt.timestamp() - t_anchor for dt in dt_objs], dtype=float)
            pred = np.zeros_like(t_rel)
            for i, c in enumerate(constituents):
                w = omegas[c]
                pred += A[i] * np.cos(w * t_rel) + B[i] * np.sin(w * t_rel)
            tide_heights = [round(float(v), 3) for v in pred.tolist()]
        except Exception:
            _LOGGER.exception("Failed to evaluate persisted tide coefficients (will return failure)")
            normalized = {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": moon_phases,
                "tide_strength": float(round(tide_strength, 3)),
                "next_high": "",
                "next_low": "",
                "next_high_height_m": None,
                "next_low_height_m": None,
                "confidence": "coefficients_invalid",
                "source": "coefficients_invalid",
                "_helpers": helpers,
            }
            self._cache = normalized
            self._last_calc = now
            return normalized

        # scalar evaluators for sampling and derivative (use python floats for reliability)
        def _tide_val(epoch_sec: float) -> float:
            rel = epoch_sec - t_anchor
            v = 0.0
            for i, c in enumerate(constituents):
                w = omegas[c]
                a = float(A[i])
                b = float(B[i])
                v += a * math.cos(w * rel) + b * math.sin(w * rel)
            return v

        def _tide_derivative(epoch_sec: float) -> float:
            rel = epoch_sec - t_anchor
            s = 0.0
            for i, c in enumerate(constituents):
                w = omegas[c]
                a = float(A[i])
                b = float(B[i])
                s += (-a * w * math.sin(w * rel) + b * w * math.cos(w * rel))
            return s

        # Small bracketed root-finder (search derivative sign changes)
        def _find_extremum_near(t_guess: float, half_window: float = 10800.0) -> Tuple[float, float]:
            a = t_guess - half_window
            b = t_guess + half_window
            steps = 24
            xs = [a + i * (b - a) / steps for i in range(steps + 1)]
            dvals = []
            for x in xs:
                try:
                    dvals.append(_tide_derivative(x))
                except Exception:
                    dvals.append(float("nan"))

            bracket = None
            for i in range(len(xs) - 1):
                fa = dvals[i]
                fb = dvals[i + 1]
                if math.isnan(fa) or math.isnan(fb):
                    continue
                if fa == 0.0:
                    bracket = (xs[i], xs[i])
                    break
                if fa * fb < 0.0:
                    bracket = (xs[i], xs[i + 1])
                    break

            if bracket is None:
                return t_guess, _tide_val(t_guess)

            aa, bb = bracket
            fa = _tide_derivative(aa)
            fb = _tide_derivative(bb)

            tol = 1e-6
            max_iter = 60
            for _ in range(max_iter):
                m = 0.5 * (aa + bb)
                fm = _tide_derivative(m)
                if abs(fm) < tol:
                    return m, _tide_val(m)
                if fa * fm <= 0:
                    bb = m
                    fb = fm
                else:
                    aa = m
                    fa = fm
            m = 0.5 * (aa + bb)
            return m, _tide_val(m)

        # Find next high/low by sampling next 48 hours on 5-minute grid, then refine
        try:
            now_epoch = now.timestamp()
            horizon_seconds = 48 * 3600.0
            sample_dt = 5 * 60.0
            n_samples = int(horizon_seconds // sample_dt) + 1
            sample_epochs = [now_epoch + i * sample_dt for i in range(n_samples)]

            # evaluate sampled values (numpy path)
            se = np.array(sample_epochs, dtype=float) - t_anchor
            sval = np.zeros_like(se)
            for i, c in enumerate(constituents):
                w = omegas[c]
                sval += float(A[i]) * np.cos(w * se) + float(B[i]) * np.sin(w * se)
            sval_list = sval.tolist()

            # find max/min indices
            max_idx = int(float(np.argmax(sval)))
            min_idx = int(float(np.argmin(sval)))
            max_epoch = sample_epochs[max_idx]
            min_epoch = sample_epochs[min_idx]
            # refine
            refined_high_epoch, refined_high_val = _find_extremum_near(max_epoch, half_window=period_seconds / 4.0)
            refined_low_epoch, refined_low_val = _find_extremum_near(min_epoch, half_window=period_seconds / 4.0)

            next_high_dt = datetime.fromtimestamp(refined_high_epoch, tz=timezone.utc)
            next_low_dt = datetime.fromtimestamp(refined_low_epoch, tz=timezone.utc)
            next_high_height = round(float(refined_high_val), 3)
            next_low_height = round(float(refined_low_val), 3)
        except Exception:
            _LOGGER.exception("Failed to compute next_high/next_low from persisted coefficients")
            next_high_dt = None
            next_low_dt = None
            next_high_height = None
            next_low_height = None

        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phases,
            "tide_strength": float(round(tide_strength, 3)),
            "next_high": next_high_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_high_dt else "",
            "next_low": next_low_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_low_dt else "",
            "next_high_height_m": next_high_height,
            "next_low_height_m": next_low_height,
            "confidence": "coefficients_applied",
            "source": "persisted_harmonic_model",
            "_helpers": helpers,
        }

        self._cache = raw_tide
        self._last_calc = now
        return raw_tide

    # ------------------ Calibration ------------------

    def _fit_coeffs_from_obs(self, X: np.ndarray, obs_arr: np.ndarray, persist: bool = False) -> Optional[np.ndarray]:
        """
        Solve least-squares; optionally persist fitted coefficients.
        """
        try:
            if obs_arr.shape[0] != X.shape[0] or X.shape[1] == 0:
                _LOGGER.debug("Observation array shape mismatch or no design columns (obs_len=%d X.shape=%s)", obs_arr.shape[0], X.shape)
                return None
            sol, *_ = np.linalg.lstsq(X, obs_arr, rcond=None)
            if sol is None:
                return None
            if persist:
                try:
                    self._fitted_coef_vec = sol.copy()
                    # persist using last helpers when available
                    constituents = None
                    t_anchor = None
                    if self._last_helpers:
                        constituents = self._last_helpers.get("constituents")
                        t_anchor = self._last_helpers.get("t_anchor")
                    self._persist_coeffs(self._fitted_coef_vec, constituents=constituents, t_anchor=t_anchor)
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
        Calibrate multi-constituent coefficients from observed heights.
        Requires at least (2 * num_constituents) samples (recommend >=6).
        Returns dict with coef_vec, residual_norm, success.
        """
        try:
            if len(obs_timestamps) != len(obs_heights):
                raise ValueError("obs_timestamps and obs_heights must have same length")

            # Ensure helpers present (this computes Skyfield helpers)
            await self.get_tide_for_timestamps(obs_timestamps)
            helpers = self._last_helpers if self._last_helpers else None
            if not helpers:
                _LOGGER.error("No helpers available for calibration; aborting")
                return None

            constituents = helpers["constituents"]
            t_anchor = float(helpers["t_anchor"])
            coef_vec_len = int(helpers["coef_vec_len"])

            # build design matrix X
            times_epoch = np.array(
                [dt_util.parse_datetime(str(ts)).replace(tzinfo=timezone.utc).timestamp() for ts in obs_timestamps],
                dtype=float,
            )
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

            # require minimum samples
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

    # ------------------ Skyfield helper: next moon transit ------------------

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

    # ------------------ compute_period_indices_for_timestamps (required by coordinator) ------------------

    async def compute_period_indices_for_timestamps(
        self,
        timestamps: Sequence[str],
        mode: str = "full_day",
        dawn_window_hours: float = 1.0,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute period -> hourly-index mapping for the provided hourly timestamps.
        This method uses Skyfield and expects timestamps as ISO strings (UTC).
        """
        # Parse into datetimes (same parsing logic used elsewhere)
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

        # Ensure Skyfield resources are loaded
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

    # ------------------ Small helpers ------------------

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


# -- module-level helpers ----

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
        # circular distance to new moon and full moon; take min
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