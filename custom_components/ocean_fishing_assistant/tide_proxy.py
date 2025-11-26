from __future__ import annotations
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncio

from homeassistant.util import dt as dt_util

# Skyfield (loaded lazily)
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield import almanac as _almanac  # type: ignore
from skyfield.framelib import ecliptic_frame
import skyfield

import numpy as np

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# numeric tolerances
# Reduced EPS_DERIV so analytic second-derivative checks are less likely to
# treat legitimate curvature as "zero". Main classification now uses a
# derivative sign test which is robust to small |second_derivative|.
EPS_DERIV = 1e-10
EPS_ROOT = 1e-9
BISECT_TOL_SEC = 1e-3  # stopping tolerance for root bisection (seconds)
GRID_SECONDS_DEFAULT = 300  # 5 minutes

# Constituent metadata (canonical)
CONSTITUENT_PERIOD_HOURS: Dict[str, float] = {
    "M2": 12.4206,
    "S2": 12.0,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
    # a few extra short/overtone constituents for better default waveform
    "P1": 24.0659,
    "Q1": 26.8683,
    "S1": 24.0,
    "M4": 12.4206 / 2.0,
    "M6": 12.4206 / 3.0,
}

# Default relative amplitudes (heuristic; not station-specific)
CONSTITUENT_DEFAULT_RATIOS: Dict[str, float] = {
    "M2": 1.00,
    "S2": 0.25,
    "N2": 0.18,
    "K1": 0.45,
    "O1": 0.25,
    "P1": 0.12,
    "Q1": 0.08,
    "S1": 0.06,
    "M4": 0.06,
    "M6": 0.02,
}


# ----
# Helpers: timestamp parsing / normalization for cache
# ----
def _to_epoch_seconds(ts: Any) -> int:
    """
    Accepts: datetime, str (ISO), numeric epoch seconds or milliseconds
    Returns: int epoch seconds (UTC)
    Raises ValueError on unrecognized input.
    """
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if isinstance(ts, float):
        return int(ts)
    if isinstance(ts, datetime):
        return int(ts.astimezone(timezone.utc).timestamp())
    if isinstance(ts, str):
        s = ts.strip()
        # numeric string?
        try:
            v = float(s)
            # Heuristic: if huge, treat as ms
            if v > 1e12:
                v = v / 1000.0
            return int(v)
        except Exception:
            pass
        # ISO parse
        try:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return int(dt.timestamp())
        except Exception as exc:
            raise ValueError(f"Unrecognized timestamp string: {ts}") from exc
    raise ValueError(f"Unsupported timestamp type: {type(ts)}")


def _normalize_timestamps_for_cache(timestamps: Sequence[Any]) -> List[int]:
    return [_to_epoch_seconds(t) for t in timestamps]


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


# ----
# Module helpers
# ----
def _compute_tide_strength_phase_heuristic(phase: Optional[float]) -> float:
    """
    Original simple phase-based heuristic preserved as fallback.
    """
    try:
        if phase is None:
            return 0.5
        p = float(phase)
        p = max(0.0, min(1.0, p))
        dist_new = min(abs(p - 0.0), abs(1.0 - p))
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        val = max(0.0, 1.0 - (dist / 0.25))
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5


def _coerce_phase(phase: Any) -> Optional[float]:
    """
    Normalize a moon phase value to 0..1 float.
    sensor.py expects tide_proxy._coerce_phase to exist.
    """
    if phase is None:
        return None
    try:
        p = float(phase)
        p = p % 1.0
        if p < 0:
            p += 1.0
        return float(p)
    except Exception:
        return None


# ----
# UTide nfactors helper (strict: no fallback)
# ----
def nfactors(jd: float, names: Sequence[str], latitude: float = 0.0) -> Dict[str, Dict[str, float]]:
    """
    Compute nodal amplitude factor (f) and phase correction (u) for the given
    Julian date (jd) and constituent names using UTide internals.

    This function is strict: it raises RuntimeError if UTide or required
    helpers/constants are not present, or KeyError if a requested constituent
    name is not found. No fallback default is returned.

    Parameters
    ----
    jd : float
        Time in days (e.g. Julian days).
    names : sequence of str
        Constituent short names, e.g. ['M2', 'S2', ...]
    latitude : float
        Latitude in degrees (passed to UTide harmonics.FUV if used)

    Returns
    ----
    dict: {name: {"f": float, "u": float_in_degrees}}
    """
    try:
        import utide  # type: ignore
        from utide import harmonics  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "UTide is required for nodal factor calculation but is not available. "
            "Install UTide in the environment so this integration can compute nodal corrections."
        ) from exc

    constit_index = getattr(utide, "constit_index_dict", None)
    if constit_index is None:
        raise RuntimeError("utide.constit_index_dict not found in installed UTide; cannot map constituent names.")

    # Ensure FUV exists
    if not hasattr(harmonics, "FUV"):
        raise RuntimeError("utide.harmonics.FUV not found in installed UTide; cannot compute nodal corrections.")

    # Map names to indices, raising if unknown or invalid
    lind_list: List[int] = []
    name_idx_map: Dict[str, int] = {}
    for nm in names:
        if nm not in constit_index:
            raise KeyError(f"Constituent '{nm}' not known to utide.constit_index_dict.")
        idx = int(constit_index[nm]) - 1
        if idx < 0:
            # Do not silently remap to zero; surface error early
            raise KeyError(f"Constituent '{nm}' maps to invalid index {idx+1} in utide.constit_index_dict.")
        lind_list.append(int(idx))
        name_idx_map[nm] = int(idx)
        _LOGGER.debug("nfactors: mapped %s -> index %d", nm, idx + 1)

    if not lind_list:
        raise RuntimeError("No constituents provided to nfactors.")

    lind = np.atleast_1d(lind_list).astype(int)

    # call harmonics.FUV; pass a 1-element time array
    t_arr = np.atleast_1d(jd)
    # ngflgs defaults (use nodal corrections)
    ngflgs = [False, False, False, False]
    F, U, V = harmonics.FUV(t_arr, jd, lind, float(latitude), ngflgs)

    F_row = np.asarray(F).reshape((1, -1))[0]
    U_row = np.asarray(U).reshape((1, -1))[0]

    out: Dict[str, Dict[str, float]] = {}
    for nm, idx in name_idx_map.items():
        pos = lind.tolist().index(int(idx))
        fval = float(F_row[pos])
        u_raw = float(U_row[pos])
        # Heuristic conversion of U to degrees: UTide historically returns cycles,
        # but some installations/versions/packaging may present radians or degrees.
        if abs(u_raw) <= 1.1:
            # likely cycles (0..1)
            uval_deg = u_raw * 360.0
        elif abs(u_raw) <= 2.0 * math.pi + 0.1:
            # likely radians
            uval_deg = math.degrees(u_raw)
        else:
            # assume degrees already
            uval_deg = u_raw
        _LOGGER.debug("nfactors: %s -> f=%s u_raw=%s u_deg=%s", nm, fval, u_raw, uval_deg)
        out[nm] = {"f": fval, "u": uval_deg}

    return out


# ----
# TideProxy
# ----
class TideProxy:
    """
    TideProxy with NO persistence. Uses in-memory deterministic coefficients (A_cos, B_sin)
    derived from CONSTITUENT_DEFAULT_RATIOS when no explicit coef_vec is supplied.

    Public methods used by the integration:
    - get_tide_for_timestamps(timestamps)
    - compute_period_indices_for_timestamps(timestamps, mode=..., dawn_window_hours=...)
    - set_coefficients(coef_vec, bias=None)
    """

    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        ttl: int = _DEFAULT_TTL,
        *,
        coef_vec: Optional[Sequence[float]] = None,
        default_m2_amp: float = 1.0,
        bias: float = 0.575626550188,
        auto_clamp_enabled: bool = False,
        min_height_floor: Optional[float] = None,
        max_amplitude_m: Optional[float] = None,
    ):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        # cache structure: {"timestamps": [epoch_ints], "raw_tide": {...}, "version": 1}
        self._cache: Optional[Dict[str, Any]] = None

        self._constituents = list(CONSTITUENT_PERIOD_HOURS.keys())
        self._bias = float(bias)
        self._auto_clamp_enabled = bool(auto_clamp_enabled)
        self._min_height_floor = None if min_height_floor is None else float(min_height_floor)
        # NOTE: _max_amplitude_m is interpreted as the HALF-RANGE amplitude (i.e. amplitude, not peak-to-peak).
        # If you prefer peak-to-peak semantics, set this accordingly or request a change.
        self._max_amplitude_m = None if max_amplitude_m is None else float(max_amplitude_m)

        # Skyfield loader (lazy)
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

        # coefficient vector (A0,B0,A1,B1,...)
        if coef_vec is not None:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size == 2 * len(self._constituents):
                self._coef_vec = arr.copy()
            else:
                _LOGGER.warning("coef_vec length mismatch; using default built-ins")
                self._coef_vec = self._build_default_coef_vec(default_m2_amp)
        else:
            # Persistent post-nodal coefficients computed from nfactors diagnostic.
            # These were produced from the station anchor and UTide nodal factors.
            # Ordering preserved: [A_M2, B_M2, A_S2, B_S2, ...]
            # If you later want to revert to heuristic defaults, set coef_vec when constructing.
            self._coef_vec = np.array(
                [
                    0.14933269066304247, 0.010009545158976542,
                    0.03765626189307662, 0.0,
                    0.02691040385059728, -0.007448362858614585,
                    0.09114141349666417, -0.006648863533063331,
                    0.040636828184277544, -0.006469123241951175,
                    0.018067509184960646, 0.00014078559909538,
                    0.01228376821279643, -0.0021918338626263174,
                    0.009033366895446204, 0.00010934048412071753,
                    0.00903734454535999, -0.002196710150478214,
                    0.0030277812797246343, -0.0006224212199091476,
                ],
                dtype=float,
            )

        # Heuristic check: warn if coefficients look like they're in cm rather than meters
        try:
            mean_abs_A = float(np.mean(np.abs(self._coef_vec[0::2])))
            if mean_abs_A > 10.0:
                _LOGGER.warning(
                    "Large mean coefficient amplitude detected (%.3f). Ensure coef_vec elements are meters (A_cos,B_sin) not cm.",
                    mean_abs_A,
                )
        except Exception:
            # non-fatal
            pass

        _LOGGER.debug(
            "TideProxy initialized lat=%s lon=%s coef_len=%d bias=%.6f clamp=%s",
            self.latitude,
            self.longitude,
            self._coef_vec.size,
            self._bias,
            self._auto_clamp_enabled,
        )

    def _build_default_coef_vec(self, m2_amp: float) -> np.ndarray:
        vals: List[float] = []
        for c in self._constituents:
            ratio = CONSTITUENT_DEFAULT_RATIOS.get(c, 0.0)
            a = float(m2_amp * ratio)
            b = 0.0
            vals.extend([a, b])
        return np.asarray(vals, dtype=float)

    def set_coefficients(self, coef_vec: Sequence[float], bias: Optional[float] = None) -> bool:
        try:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size != 2 * len(self._constituents):
                _LOGGER.error(
                    "set_coefficients: coef_vec length %d != expected %d", arr.size, 2 * len(self._constituents)
                )
                return False
            self._coef_vec = arr.copy()
            if bias is not None:
                self._bias = float(bias)
            self._cache = None
            # Heuristic check here too
            try:
                mean_abs_A = float(np.mean(np.abs(self._coef_vec[0::2])))
                if mean_abs_A > 10.0:
                    _LOGGER.warning(
                        "Large mean coefficient amplitude detected in set_coefficients (%.3f). Ensure coef_vec elements are meters (A_cos,B_sin) not cm.",
                        mean_abs_A,
                    )
            except Exception:
                pass
            _LOGGER.info("set_coefficients applied (len=%d) bias=%.6f", arr.size, self._bias)
            return True
        except Exception:
            _LOGGER.exception("set_coefficients failed")
            return False

    async def _ensure_loaded(self) -> None:
        if self._sf_eph is not None and self._sf_ts is not None:
            return
        async with self._load_lock:
            if self._sf_eph is not None and self._sf_ts is not None:
                return

            def _blocking_load():
                sf_ts = self._loader.timescale()
                sf_eph = self._loader("de421.bsp")
                sf_wgs = wgs84
                sf_almanac = _almanac
                version = getattr(skyfield, "__version__", "unknown")
                return sf_ts, sf_eph, sf_wgs, sf_almanac, version

            try:
                sf_ts, sf_eph, sf_wgs, sf_almanac, version = await self.hass.async_add_executor_job(
                    _blocking_load
                )
                self._sf_ts = sf_ts
                self._sf_eph = sf_eph
                self._sf_wgs = sf_wgs
                self._sf_almanac = sf_almanac
                _LOGGER.info("Skyfield loaded version=%s", version)
            except Exception:
                _LOGGER.exception("Failed to load Skyfield resources")
                raise

    async def get_tide_for_timestamps(self, timestamps: Sequence[Any]) -> Dict[str, Any]:
        """
        Main entrypoint: timestamps can be ISO strings, epoch numbers, or datetime objects.
        Returns a strict dict with tide heights and next_high/next_low objects.
        """
        now = dt_util.now().astimezone(timezone.utc)

        # Early-guard for empty timestamps
        if not timestamps:
            return {
                "timestamps": [],
                "tide_height_m": [],
                "tide_phase": [],
                "tide_strength": 0.0,
                "confidence": "no_timestamps",
                "source": "tide_proxy",
                "next_high": None,
                "next_low": None,
            }

        # Normalize incoming timestamps for cache compare (epoch ints)
        try:
            new_keys = _normalize_timestamps_for_cache(timestamps)
        except Exception:
            # fallback: attempt per-element parse using dt_util, but fail if any unparseable
            dt_objs_try: List[datetime] = []
            for ts in timestamps:
                parsed = dt_util.parse_datetime(str(ts))
                if parsed is None:
                    try:
                        v = float(ts)
                        if v > 1e12:
                            v = v / 1000.0
                        parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                    except Exception:
                        parsed = None
                if parsed is None:
                    _LOGGER.error("Unable to parse timestamp for cache normalization: %s", ts)
                    return {
                        "timestamps": [str(t) for t in timestamps],
                        "tide_height_m": [None] * len(timestamps),
                        "confidence": "bad_timestamps",
                        "source": "tide_proxy",
                    }
                dt_objs_try.append(parsed.astimezone(timezone.utc))
            new_keys = [int(dt.timestamp()) for dt in dt_objs_try]

        # cache fast-path (compare normalized epoch lists)
        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached_keys = self._cache.get("timestamps")
            if cached_keys is not None and cached_keys == new_keys:
                return self._cache["raw_tide"]

        # parse timestamps into aware datetimes (UTC) in original order
        dt_objs: List[datetime] = []
        for ts in timestamps:
            # If ts already a datetime, keep
            if isinstance(ts, datetime):
                dt = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                dt_objs.append(dt)
                continue
            # numeric string / number
            try:
                epoch = None
                if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.strip().replace(".", "", 1).isdigit()):
                    # numeric-ish
                    v = float(ts)
                    if v > 1e12:
                        v = v / 1000.0
                    epoch = int(v)
                    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                    dt_objs.append(dt)
                    continue
            except Exception:
                pass

            # fallback to ISO parsing
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
                return {
                    "timestamps": [str(t) for t in timestamps],
                    "tide_height_m": [None] * len(timestamps),
                    "confidence": "bad_timestamps",
                    "source": "tide_proxy",
                }
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)

        # ensure skyfield to compute helpers (moon phase/anchor)
        try:
            await self._ensure_loaded()
        except Exception:
            _LOGGER.exception("Skyfield unavailable")
            return {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "confidence": "astronomical_unavailable",
                "source": "astronomical_unavailable",
            }

        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        # anchor: try moon transit, else first timestamp with a longitude shift
        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            moon_transit_dt = None

        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR
        if moon_transit_dt is None:
            lon_shift = (self.longitude / 360.0) * period_seconds
            t_anchor = anchor_epoch - lon_shift
        else:
            lon_shift = 0.0
            t_anchor = anchor_epoch

        # build omegas
        periods_sec = {k: (CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in self._constituents}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        # evaluate model using in-memory coef_vec (order preserved)
        coef_arr = np.asarray(self._coef_vec, dtype=float)
        A = coef_arr[0::2].astype(float)
        B = coef_arr[1::2].astype(float)

        # Save copies for diagnostics before nodal correction
        A_orig = A.copy()
        B_orig = B.copy()

        # --- Apply UTide nodal corrections (STRICT: will raise if UTide missing or mismatch) ---
        # convert t_anchor (epoch seconds) to Julian-like days for UTide:
        jd_anchor = float(t_anchor) / 86400.0 + 2440587.5
        _LOGGER.debug("Nodal correction context: t_anchor=%s jd_anchor=%s lon_shift=%s", t_anchor, jd_anchor, lon_shift)

        # Run potentially-blocking UTide import and computation in executor to avoid blocking the event loop
        try:
            nf = await self.hass.async_add_executor_job(nfactors, jd_anchor, self._constituents, float(self.latitude))
        except Exception:
            _LOGGER.exception("Failed to compute nodal factors via UTide")
            raise

        # convert A,B to amplitude/phase, apply f (multiplier) and u (degrees), then back
        for i, cname in enumerate(self._constituents):
            if cname not in nf:
                raise KeyError(f"nfactors did not return entry for constituent '{cname}'")
            fval = float(nf[cname].get("f", 1.0))
            udeg = float(nf[cname].get("u", 0.0))
            R = math.hypot(float(A[i]), float(B[i]))
            if R <= 0.0 or not np.isfinite(R):
                continue
            phi = math.atan2(float(B[i]), float(A[i]))  # radians
            R2 = R * float(fval)
            phi2 = phi + math.radians(float(udeg))
            A[i] = float(R2 * math.cos(phi2))
            B[i] = float(R2 * math.sin(phi2))

        # diagnostic logging: pre/post nodal amplitude stats
        try:
            Rs_pre = np.hypot(A_orig, B_orig)
            Rs_post = np.hypot(A, B)
            _LOGGER.debug(
                "constituent amps: pre mean=%.4f max=%.4f post mean=%.4f max=%.4f",
                float(Rs_pre.mean()),
                float(Rs_pre.max()),
                float(Rs_post.mean()),
                float(Rs_post.max()),
            )
        except Exception:
            # non-fatal
            pass

        # compute prediction in original timestamp order
        t_rel = np.array([dt.timestamp() - t_anchor for dt in dt_objs], dtype=float)
        pred = np.zeros_like(t_rel)
        for i, c in enumerate(self._constituents):
            w = omegas[c]
            pred += A[i] * np.cos(w * t_rel) + B[i] * np.sin(w * t_rel)
        if float(self._bias) != 0.0:
            pred = pred + float(self._bias)

        # clamp/scale (optional)
        if self._auto_clamp_enabled:
            try:
                if self._max_amplitude_m is not None:
                    # Interpret configured _max_amplitude_m as HALF-RANGE amplitude (peak from mean)
                    current_half = (float(np.max(pred)) - float(np.min(pred))) / 2.0
                    if current_half > float(self._max_amplitude_m) and current_half > 1e-12:
                        mean_val = float(np.mean(pred))
                        scale = float(self._max_amplitude_m) / current_half
                        pred = mean_val + (pred - mean_val) * scale
                if self._min_height_floor is not None:
                    pred = np.maximum(pred, float(self._min_height_floor))
            except Exception:
                _LOGGER.exception("Error applying clamp/scale")

        tide_heights = [round(float(v), 3) for v in pred.tolist()]

        # --- compute next high/low (timestamps + heights) for sensor compatibility (strict dict shape) ---
        next_high: Optional[str] = None
        next_low: Optional[str] = None
        next_high_height: Optional[float] = None
        next_low_height: Optional[float] = None

        try:
            now_ts = now.timestamp()

            # use sorted times for analytic root finding (safer)
            t_epochs = np.array([dt.timestamp() for dt in dt_objs], dtype=float)
            order = np.argsort(t_epochs)
            t_sorted = t_epochs[order]
            pred_sorted = pred[order]

            # find first future index in sorted array
            n_sorted = len(t_sorted)
            first_future_idx_sorted = n_sorted
            for i_s, tval in enumerate(t_sorted):
                if tval >= now_ts:
                    first_future_idx_sorted = i_s
                    break

            # helpers for continuous model evaluation
            def _height_at_epoch(epoch_ts: float) -> float:
                t_rel_local = epoch_ts - t_anchor
                s = float(self._bias) if hasattr(self, "_bias") else 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += float(A[j]) * math.cos(w * t_rel_local) + float(B[j]) * math.sin(w * t_rel_local)
                return s

            def _derivative_at_epoch(epoch_ts: float) -> float:
                t_rel_local = epoch_ts - t_anchor
                s = 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += -float(A[j]) * w * math.sin(w * t_rel_local) + float(B[j]) * w * math.cos(w * t_rel_local)
                return s

            def _second_derivative_at_epoch(epoch_ts: float) -> float:
                t_rel_local = epoch_ts - t_anchor
                s = 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += -float(A[j]) * (w ** 2) * math.cos(w * t_rel_local) - float(B[j]) * (w ** 2) * math.sin(w * t_rel_local)
                return s

            def _find_root_bisection(f, a: float, b: float, maxiter: int = 60, tol: float = BISECT_TOL_SEC) -> Optional[float]:
                fa = f(a)
                fb = f(b)
                if abs(fa) < EPS_ROOT:
                    return a
                if abs(fb) < EPS_ROOT:
                    return b
                if fa * fb > 0:
                    return None
                lo = a
                hi = b
                fa_local = fa
                for _ in range(maxiter):
                    mid = 0.5 * (lo + hi)
                    fm = f(mid)
                    if abs(fm) < EPS_ROOT or (hi - lo) < tol:
                        return mid
                    if fa_local * fm <= 0:
                        hi = mid
                    else:
                        lo = mid
                        fa_local = fm
                return 0.5 * (lo + hi)

            # --- Robust derivative scan (vectorized over a uniform grid) ---
            candidates: List[float] = []

            scan_start = max(now_ts, float(t_sorted[0])) if t_sorted.size > 0 else now_ts
            scan_end = float(t_sorted[-1]) if t_sorted.size > 0 else now_ts

            if scan_end > scan_start:
                step = float(GRID_SECONDS_DEFAULT)
                grid = np.arange(scan_start, scan_end + 0.5 * step, step, dtype=float)
                # relative times
                t_rel_grid = grid - t_anchor
                # vectorized derivative evaluation
                d_grid = np.zeros_like(t_rel_grid)
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    Aj = float(A[j])
                    Bj = float(B[j])
                    # vectorized contribution
                    d_grid += -Aj * w * np.sin(w * t_rel_grid) + Bj * w * np.cos(w * t_rel_grid)

                # near-zero derivative points as direct candidates
                near_zero_idx = np.where(np.abs(d_grid) < EPS_DERIV)[0]
                for idx in near_zero_idx:
                    t_candidate = float(grid[idx])
                    if t_candidate >= now_ts:
                        candidates.append(t_candidate)

                # sign-change brackets and bisection refinement
                prod = d_grid[:-1] * d_grid[1:]
                sign_change_idx = np.where(prod < 0)[0]
                for idx in sign_change_idx:
                    a = float(grid[idx])
                    b = float(grid[idx + 1])
                    root = _find_root_bisection(_derivative_at_epoch, a, b)
                    if root is not None and root >= now_ts:
                        candidates.append(root)

            # Deduplicate & sort
            candidates = sorted(set([float(x) for x in candidates]))

            # --- classify candidates into maxima/minima robustly using derivative sign test ---
            maxima: List[Tuple[float, float]] = []
            minima: List[Tuple[float, float]] = []

            # delta for finite-difference check: fraction of grid step (but at least 1s)
            delta_sec = max(1.0, float(GRID_SECONDS_DEFAULT) / 8.0)

            classification_debug: List[Tuple[str, float, float, float, str]] = []

            for rt in candidates:
                try:
                    # pick probe points safely inside the scan window
                    t_before = rt - delta_sec
                    t_after = rt + delta_sec

                    # clamp probes to scan boundaries if needed
                    t_before = max(t_before, scan_start)
                    t_after = min(t_after, scan_end)

                    d_before = _derivative_at_epoch(t_before)
                    d_after = _derivative_at_epoch(t_after)

                    cls = ""
                    # classification by sign change: d_before > 0 && d_after < 0 => maximum
                    #                            d_before < 0 && d_after > 0 => minimum
                    if d_before > 0.0 and d_after < 0.0:
                        maxima.append((rt, _height_at_epoch(rt)))
                        cls = "max_sign"
                    elif d_before < 0.0 and d_after > 0.0:
                        minima.append((rt, _height_at_epoch(rt)))
                        cls = "min_sign"
                    else:
                        # If sign test ambiguous (flat or numeric), fall back to analytic second derivative
                        sec = _second_derivative_at_epoch(rt)
                        if sec < -EPS_DERIV:
                            maxima.append((rt, _height_at_epoch(rt)))
                            cls = "max_sec"
                        elif sec > EPS_DERIV:
                            minima.append((rt, _height_at_epoch(rt)))
                            cls = "min_sec"
                        else:
                            cls = "ambiguous"

                    if len(classification_debug) < 12:
                        # reuse cached 'sec' rather than calling the function again
                        sec_val = sec if 'sec' in locals() else _second_derivative_at_epoch(rt)
                        classification_debug.append(
                            (
                                datetime.fromtimestamp(rt, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                                float(d_before),
                                float(d_after),
                                float(sec_val),
                                cls,
                            )
                        )
                except Exception as e:
                    # log for debugging but continue processing other candidates
                    _LOGGER.debug("candidate classification failed at %s: %s", rt, e, exc_info=True)
                    continue

            # debug dump: log first few candidates and classifications
            try:
                _LOGGER.debug(
                    "next-extrema classify: scan_start=%s scan_end=%s grid_step=%ss classification_debug(sample)=%s",
                    datetime.fromtimestamp(scan_start, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    datetime.fromtimestamp(scan_end, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    GRID_SECONDS_DEFAULT,
                    classification_debug,
                )
            except Exception:
                # keep silent (best-effort logging)
                pass

            # pick earliest future max/min
            next_high_tuple: Optional[Tuple[float, float]] = None
            next_low_tuple: Optional[Tuple[float, float]] = None
            if maxima:
                next_high_tuple = min(maxima, key=lambda x: x[0])
            if minima:
                next_low_tuple = min(minima, key=lambda x: x[0])

            # Extremely small local fallback: refine near first future sample interval if nothing found
            if next_high_tuple is None or next_low_tuple is None:
                if first_future_idx_sorted < t_sorted.size:
                    # bracket small window around first future sample (one interval)
                    a_idx = max(0, first_future_idx_sorted - 1)
                    b_idx = min(n_sorted - 1, first_future_idx_sorted + 1)
                    a = float(t_sorted[a_idx])
                    b = float(t_sorted[b_idx])
                    if b > a:
                        root = _find_root_bisection(_derivative_at_epoch, a, b)
                        if root is not None and root >= now_ts:
                            h = _height_at_epoch(root)
                            sec = _second_derivative_at_epoch(root)
                            if sec < -EPS_DERIV and next_high_tuple is None:
                                next_high_tuple = (root, h)
                            elif sec > EPS_DERIV and next_low_tuple is None:
                                next_low_tuple = (root, h)

            # If still missing (very unlikely), fall back to sampled argmax/argmin on sorted future samples
            if next_high_tuple is None:
                rel = pred_sorted[first_future_idx_sorted:] if first_future_idx_sorted < pred_sorted.size else np.array([])
                if rel.size:
                    idx_rel = int(np.argmax(rel))
                    idx_sorted = first_future_idx_sorted + idx_rel
                    next_high_tuple = (float(t_sorted[idx_sorted]), float(pred_sorted[idx_sorted]))
            if next_low_tuple is None:
                rel = pred_sorted[first_future_idx_sorted:] if first_future_idx_sorted < pred_sorted.size else np.array([])
                if rel.size:
                    idx_rel = int(np.argmin(rel))
                    idx_sorted = first_future_idx_sorted + idx_rel
                    next_low_tuple = (float(t_sorted[idx_sorted]), float(pred_sorted[idx_sorted]))

            if next_high_tuple is not None:
                nh_ts, nh_h = next_high_tuple
                next_high = datetime.fromtimestamp(nh_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                next_high_height = float(round(float(nh_h), 3))
            if next_low_tuple is not None:
                nl_ts, nl_h = next_low_tuple
                next_low = datetime.fromtimestamp(nl_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                next_low_height = float(round(float(nl_h), 3))
        except Exception:
            _LOGGER.debug("Failed to compute next_high/next_low", exc_info=True)

        # moon phases for payload (best-effort)
        try:
            earth = sf_eph["earth"]
            sun_obj = sf_eph["sun"]
            moon_obj = sf_eph["moon"]
            times_list = [sf_ts.from_datetime(dt) for dt in dt_objs]
            moon_phases: List[Optional[float]] = []
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

        # compute tide_strength: prefer Skyfield physical proxy when available, else fallback heuristic
        tide_strength_value = 0.5
        try:
            if sf_ts is not None and sf_eph is not None and len(dt_objs) > 0:
                # try to compute a physical proxy using moon distance and alignment
                t0 = sf_ts.from_datetime(dt_objs[0])
                earth = sf_eph["earth"]
                moon_obj = sf_eph["moon"]
                sun_obj = sf_eph["sun"]
                mpos = earth.at(t0).observe(moon_obj)
                spos = earth.at(t0).observe(sun_obj)
                d_moon_au = float(mpos.distance().au)
                moon_ecl = mpos.apparent().frame_latlon(ecliptic_frame)
                sun_ecl = spos.apparent().frame_latlon(ecliptic_frame)
                lon_moon = float(moon_ecl[1].radians)
                lon_sun = float(sun_ecl[1].radians)
                delta = (lon_moon - lon_sun)  # radians
                # heuristic physical proxy: alignment term * inverse-cube distance
                d_ref = 0.00257  # approx mean moon distance (AU)
                # compute raw strength (0..~1.2)
                raw = 0.5 * (1.0 + math.cos(delta)) * (d_ref / max(d_moon_au, 1e-9)) ** 3
                # normalize by plausible maximum (perigee spring)
                d_min_plausible = 0.0024
                max_raw = 0.5 * (1.0 + 1.0) * (d_ref / d_min_plausible) ** 3  # = (d_ref/d_min)^3
                strength = raw / max_raw if max_raw > 0 else raw
                tide_strength_value = float(max(0.0, min(1.0, strength)))
            else:
                tide_strength_value = float(_compute_tide_strength_phase_heuristic(_coerce_phase(moon_phases[0] if moon_phases else None)))
        except Exception as e:
            # Only fall back to phase heuristic when the physical proxy computation fails.
            _LOGGER.debug("Physical tide_strength proxy failed, falling back to phase heuristic: %s", e, exc_info=True)
            tide_strength_value = float(_compute_tide_strength_phase_heuristic(_coerce_phase(moon_phases[0] if moon_phases else None)))

        # Build strict dicts for next_high/next_low (sensor expects dict shape)
        next_high_obj = None
        next_low_obj = None
        if next_high is not None:
            try:
                next_high_obj = {"timestamp": next_high, "height_m": next_high_height}
            except Exception:
                next_high_obj = None
        if next_low is not None:
            try:
                next_low_obj = {"timestamp": next_low, "height_m": next_low_height}
            except Exception:
                next_low_obj = None

        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phases,
            "tide_strength": float(round(tide_strength_value, 3)),
            "confidence": "in_memory_model",
            "source": "in_memory_harmonic_model",
            "_helpers": {
                "constituents": self._constituents,
                "t_anchor": float(t_anchor),
                "period_seconds": float(period_seconds),
                "coef_vec_len": int(self._coef_vec.size),
            },
            # Strict canonical tide objects expected by sensor.py:
            "next_high": next_high_obj,
            "next_low": next_low_obj,
        }

        # store cache normalized timestamps (epoch ints) and raw_tide payload
        try:
            self._cache = {"timestamps": new_keys, "raw_tide": raw_tide, "version": 1}
        except Exception:
            self._cache = {"timestamps": new_keys, "raw_tide": raw_tide}
        self._last_calc = now
        return raw_tide

    async def _async_find_next_moon_transit(self, sf_eph, sf_ts, sf_almanac, sf_wgs, start_dt: datetime) -> Optional[datetime]:
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

    async def compute_period_indices_for_timestamps(
        self,
        timestamps: Sequence[Any],
        mode: str = "full_day",
        dawn_window_hours: float = 1.0,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute period indices mapping, used by the coordinator and formatter.
        """
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

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def _iso_z_local(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")

        for d in dates_needed:
            try:
                day_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)

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
                    for t, ev in zip(times, events):
                        try:
                            evt_dt = t.utc_datetime().replace(tzinfo=timezone.utc)
                        except Exception:
                            evt_dt = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
                        if bool(ev):
                            sunrise_candidates.append(evt_dt)
                        else:
                            sunset_candidates.append(evt_dt)
                    if not sunrise_candidates or not sunset_candidates:
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
                    result[date_key]["dawn"] = {"indices": dawn_indices, "start": _iso_z_local(dawn_start), "end": _iso_z_local(dawn_end)}
                    result[date_key]["dusk"] = {"indices": dusk_indices, "start": _iso_z_local(dusk_start), "end": _iso_z_local(dusk_end)}
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
                        result[date_key][pname] = {"indices": indices, "start": _iso_z_local(pstart), "end": _iso_z_local(pend)}
            except Exception as exc:
                _LOGGER.exception("compute_period_indices_for_timestamps failed for date %s: %s", d.isoformat(), exc)
                raise
        return result