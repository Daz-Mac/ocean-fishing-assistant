# tide_proxy.py
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

# UTide will be imported lazily in an executor to avoid blocking the HA event loop.
utide = None
_utide_utils = None
_utide_harmonics = None
_UTIDE_IMPORTED = False

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# numeric tolerances
EPS_DERIV = 1e-8
EPS_ROOT = 1e-9
BISECT_TOL_SEC = 1e-3  # stopping tolerance for root bisection (seconds)
GRID_SECONDS_DEFAULT = 300  # 5 minutes

# Reference epoch for amplitude/phase storage (UTC)
T_REF = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T_REF_EPOCH = T_REF.timestamp()

# Vertical datum offset (meters). Set per-station by changing instance attribute if needed.
VERTICAL_DATUM_OFFSET_M = 0.0

# Constituent metadata (canonical)
# Kept to ~10 constituents useful for short-term forecasts (today + 5 days)
CONSTITUENT_PERIOD_HOURS: Dict[str, float] = {
    "M2": 12.420641,  # principal lunar semidiurnal
    "S2": 12.0,       # principal solar semidiurnal
    "N2": 12.658348,  # larger lunar elliptic semidiurnal
    "K1": 23.934472,  # lunisolar diurnal
    "O1": 25.819338,  # lunar diurnal
    "P1": 24.065887,
    "Q1": 26.868352,
    "S1": 24.0,
    "M4": 12.420641 / 2.0,
    "M6": 12.420641 / 3.0,
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


def _ensure_amplitude_meters(amplitude: float, name: Optional[str] = None) -> float:
    """
    Heuristic guard to ensure amplitude is in meters.

    - If amplitude > 50 -> likely millimetres (mm) -> convert /1000
    - If amplitude between 1 and 50 -> ambiguous (could be cm or m) -> assume meters.
    Adjust heuristics if you expect different source units.
    """
    try:
        a = float(amplitude)
    except Exception:
        return float(amplitude)
    if a > 50.0:
        a_m = a / 1000.0
        _LOGGER.debug("Converted amplitude for %s from likely mm: %s -> %s m", name or "<unknown>", a, a_m)
        return a_m
    return a


def amp_phase_to_coefvec(
    amplitudes: Sequence[float],
    phases: Sequence[float],
    constituents: Sequence[str],
    omegas: Dict[str, float],
    t_ref_epoch: float,
    t_anchor_epoch: float,
) -> np.ndarray:
    """
    Convert amplitude/phase pairs (H_k, g_k) defined at reference epoch t_ref into
    coef_vec (A0,B0,A1,B1,...) aligned with t_anchor used by the internal model.

    phases are expected in radians.
    Formula:
      phi = g_k + omega * (t_ref - t_anchor)
      term = H_k * cos(omega*(t - t_anchor) + phi)
      => A = H_k * cos(phi), B = -H_k * sin(phi)
    """
    amps = np.asarray(amplitudes, dtype=float)
    phs = np.asarray(phases, dtype=float)
    vals: List[float] = []
    for i, c in enumerate(constituents):
        H = float(amps[i])
        g = float(phs[i])
        omega = float(omegas[c])
        phi = g + omega * (t_ref_epoch - t_anchor_epoch)
        A = H * math.cos(phi)
        B = -H * math.sin(phi)
        vals.extend([A, B])
    return np.asarray(vals, dtype=float)


def coefvec_to_amp_phase(
    coef_vec: Sequence[float],
    constituents: Sequence[str],
    omegas: Dict[str, float],
    t_ref_epoch: float,
    t_anchor_epoch: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert internal coef_vec (A,B ordered) back to amplitude & phase at t_ref.
    Returns (H_array, g_array) with g in radians.
    """
    arr = np.asarray(coef_vec, dtype=float)
    A = arr[0::2]
    B = arr[1::2]
    Hs = []
    gs = []
    for i, c in enumerate(constituents):
        omega = float(omegas[c])
        A_i = float(A[i])
        B_i = float(B[i])
        H = math.hypot(A_i, B_i)
        phi = math.atan2(-B_i, A_i)  # because A = H cos(phi), B = -H sin(phi)
        # convert phi at t_anchor back to phase at t_ref: g = phi - omega*(t_ref - t_anchor)
        g = phi - omega * (t_ref_epoch - t_anchor_epoch)
        Hs.append(H)
        gs.append(g)
    return np.asarray(Hs, dtype=float), np.asarray(gs, dtype=float)


async def _ensure_utide_loaded(hass) -> None:
    """
    Lazily import UTide in an executor (non-blocking to the event loop),
    verify utilities.nfactors exists (strict requirement), and set module globals.
    Raises RuntimeError if import or API checks fail (strict behavior).
    """
    global utide, _utide_utils, _utide_harmonics, _UTIDE_IMPORTED

    if _UTIDE_IMPORTED:
        return

    def _blocking_import():
        # executed in executor thread
        import importlib
        _ut = importlib.import_module("utide")
        _utils = getattr(_ut, "utilities", None)
        _harm = getattr(_ut, "harmonics", None)
        ver = getattr(_ut, "__version__", None)
        return _ut, _utils, _harm, ver

    try:
        _ut, _utils, _harm, ver = await hass.async_add_executor_job(_blocking_import)
    except Exception as exc:
        raise RuntimeError(
            "UTide (>=0.3.1) is required for nodal corrections. "
            "Install it in your Home Assistant environment (e.g. `pip install utide==0.3.1`) and restart."
        ) from exc

    # Strict mode: require utilities.nfactors to be present
    if not (_utils and hasattr(_utils, "nfactors")):
        raise RuntimeError(
            "UTide strict mode: required function utilities.nfactors missing. "
            "Install utide==0.3.1 (or compatible) and restart Home Assistant."
        )

    # assign module globals
    utide = _ut
    _utide_utils = _utils
    _utide_harmonics = _harm
    _UTIDE_IMPORTED = True
    _LOGGER.debug("UTide successfully imported (lazy) and utilities.nfactors verified")


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
        bias: float = 0.0,
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
        self._max_amplitude_m = None if max_amplitude_m is None else float(max_amplitude_m)

        # allow per-instance vertical datum override
        self.VERTICAL_DATUM_OFFSET_M = VERTICAL_DATUM_OFFSET_M

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
            self._coef_vec = self._build_default_coef_vec(default_m2_amp)

        _LOGGER.debug(
            "TideProxy initialized lat=%s lon=%s coef_len=%d bias=%.3f clamp=%s",
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
            a = _ensure_amplitude_meters(a, name=c)
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
            _LOGGER.info("set_coefficients applied (len=%d) bias=%.3f", arr.size, self._bias)
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

    def _synthesize_grid(self, omegas: Dict[str, float], A: np.ndarray, B: np.ndarray, t_anchor: float, start_epoch: float, end_epoch: float, grid_seconds: int = GRID_SECONDS_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize the harmonic model onto a regular grid (epoch seconds) between start_epoch and end_epoch.
        Returns (grid_epochs, grid_pred)
        """
        if end_epoch <= start_epoch:
            return np.array([], dtype=float), np.array([], dtype=float)
        grid_epochs = np.arange(start_epoch, end_epoch + 1, grid_seconds, dtype=float)
        t_rel = grid_epochs - t_anchor
        pred = np.zeros_like(grid_epochs, dtype=float)
        for i, c in enumerate(self._constituents):
            w = omegas[c]
            pred += float(A[i]) * np.cos(w * t_rel) + float(B[i]) * np.sin(w * t_rel)
        if float(self._bias) != 0.0:
            pred = pred + float(self._bias)
        # Apply vertical datum offset (per-instance)
        if getattr(self, "VERTICAL_DATUM_OFFSET_M", 0.0):
            pred = pred + float(self.VERTICAL_DATUM_OFFSET_M)
        return grid_epochs, pred

    async def get_tide_for_timestamps(self, timestamps: Sequence[Any]) -> Dict[str, Any]:
        """
        Main entrypoint: timestamps can be ISO strings, epoch numbers, or datetime objects.
        Returns a strict dict with tide heights and next_high/next_low objects.
        """
        now = dt_util.now().astimezone(timezone.utc)

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
            t_anchor = anchor_epoch

        # build omegas
        periods_sec = {k: (CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in self._constituents}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        # evaluate model using in-memory coef_vec (order preserved)
        coef_arr = np.asarray(self._coef_vec, dtype=float)
        A = coef_arr[0::2].astype(float)
        B = coef_arr[1::2].astype(float)

        # Try to apply nodal corrections using UTide (strict, lazy non-blocking import)
        try:
            # Ensure UTide is imported and API verified (async, non-blocking)
            await _ensure_utide_loaded(self.hass)

            Hs, gs = coefvec_to_amp_phase(self._coef_vec, self._constituents, omegas, float(T_REF_EPOCH), float(t_anchor))
            # Ensure amplitudes reasonable (meters)
            Hs = np.array([_ensure_amplitude_meters(h, name=c) for h, c in zip(Hs, self._constituents)], dtype=float)
            gs = np.asarray(gs, dtype=float)

            # compute Julian date for t_anchor (UTC)
            t_jd = float(t_anchor) / 86400.0 + 2440587.5

            # Strict nodal-factor discovery: require utilities.nfactors only (no fallbacks)
            f_dict = {}

            # Must have utide utilities.nfactors present (strict requirement)
            if not (_utide_utils and hasattr(_utide_utils, "nfactors")):
                raise RuntimeError(
                    "UTide strict mode: required function utilities.nfactors missing. "
                    "Install utide==0.3.1 (or compatible) and restart Home Assistant."
                )

            # Call nfactors and validate its output strictly
            try:
                nf_out = _utide_utils.nfactors(t_jd, self._constituents)
            except Exception as exc:
                raise RuntimeError("UTide strict mode: utilities.nfactors() call failed") from exc

            if not isinstance(nf_out, dict):
                raise RuntimeError("UTide strict mode: utilities.nfactors() returned unexpected type (expected dict)")

            # Expect each constituent to be present with either tuple/list (f,u) or dict with 'f' and 'u'
            for name in self._constituents:
                if name not in nf_out:
                    raise RuntimeError(f"UTide strict mode: nfactors did not return data for constituent '{name}'")
                val = nf_out[name]
                if isinstance(val, (tuple, list)) and len(val) >= 2:
                    f_val = float(val[0])
                    u_deg = float(val[1])
                elif isinstance(val, dict):
                    if "f" not in val or "u" not in val:
                        raise RuntimeError(f"UTide strict mode: nfactors returned dict for '{name}' missing 'f' or 'u'")
                    f_val = float(val["f"])
                    u_deg = float(val["u"])
                else:
                    raise RuntimeError(
                        f"UTide strict mode: nfactors returned unsupported format for '{name}': {type(val)}"
                    )
                f_dict[name] = (f_val, math.radians(u_deg))

            # apply nodal corrections
            Hs_corr = []
            gs_corr = []
            for name, H_val, g_val in zip(self._constituents, Hs, gs):
                f_u = f_dict.get(name)
                if f_u:
                    f_val, u_rad = f_u
                    Hc = float(H_val) * float(f_val)
                    gc = float(g_val) + float(u_rad)
                    Hs_corr.append(Hc)
                    gs_corr.append(gc)
                else:
                    # In strict mode we shouldn't ever reach here because we required each constituent above
                    Hs_corr.append(float(H_val))
                    gs_corr.append(float(g_val))
            Hs = np.asarray(Hs_corr, dtype=float)
            gs = np.asarray(gs_corr, dtype=float)
            # rebuild coef vec at t_anchor
            new_coef = amp_phase_to_coefvec(Hs, gs, self._constituents, omegas, float(T_REF_EPOCH), float(t_anchor))
            coef_arr = np.asarray(new_coef, dtype=float)
            A = coef_arr[0::2].astype(float)
            B = coef_arr[1::2].astype(float)
            _LOGGER.debug("Applied UTide nodal corrections for t_anchor=%s (jd=%s)", t_anchor, t_jd)
        except Exception:
            _LOGGER.exception("UTide nodal correction failed (strict mode) - aborting initialization")
            raise

        # For robust short-term forecasts, synthesize a fine grid over today + 5 days and use it for interpolation and peak finding
        grid_start_dt = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
        grid_end_dt = grid_start_dt + timedelta(days=5)
        start_epoch = grid_start_dt.timestamp()
        end_epoch = grid_end_dt.timestamp()

        grid_epochs, grid_pred = self._synthesize_grid(omegas, A, B, t_anchor, start_epoch, end_epoch, GRID_SECONDS_DEFAULT)

        # If clamp/scale enabled, apply to grid and (later) to sampled preds
        if self._auto_clamp_enabled and grid_pred.size:
            try:
                if self._max_amplitude_m is not None:
                    current_amp = float(np.max(grid_pred) - np.min(grid_pred))
                    if current_amp > float(self._max_amplitude_m) and current_amp > 1e-12:
                        mean_val = float(np.mean(grid_pred))
                        scale = float(self._max_amplitude_m) / current_amp
                        grid_pred = mean_val + (grid_pred - mean_val) * scale
                if self._min_height_floor is not None:
                    grid_pred = np.maximum(grid_pred, float(self._min_height_floor))
            except Exception:
                _LOGGER.exception("Error applying clamp/scale to grid")

        # compute heights for requested timestamps by interpolating the grid where possible
        req_epochs = np.array([dt.timestamp() for dt in dt_objs], dtype=float)
        tide_heights_out: List[Optional[float]] = []
        for e in req_epochs:
            if grid_epochs.size and e >= grid_epochs[0] and e <= grid_epochs[-1]:
                h = float(np.interp(e, grid_epochs, grid_pred))
            else:
                # fallback to analytic evaluation for out-of-grid times
                t_rel_single = e - t_anchor
                s = float(self._bias) if hasattr(self, "_bias") else 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += float(A[j]) * math.cos(w * t_rel_single) + float(B[j]) * math.sin(w * t_rel_single)
                # apply vertical datum offset if not applied in _synthesize_grid path
                if getattr(self, "VERTICAL_DATUM_OFFSET_M", 0.0):
                    s = s + float(self.VERTICAL_DATUM_OFFSET_M)
                h = float(s)
            tide_heights_out.append(round(h, 3))

        tide_heights = tide_heights_out

        # --- compute next high/low using the grid for candidate detection and analytic refinement ---
        next_high: Optional[str] = None
        next_low: Optional[str] = None
        next_high_height: Optional[float] = None
        next_low_height: Optional[float] = None

        try:
            now_ts = now.timestamp()
            # candidate detection via derivative sampled on grid (analytic derivative evaluated on grid points)
            candidates: List[float] = []
            if grid_epochs.size:
                # analytic derivative at grid points
                t_rel_grid = grid_epochs - t_anchor
                deriv_grid = np.zeros_like(grid_pred)
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    deriv_grid += -float(A[j]) * w * np.sin(w * t_rel_grid) + float(B[j]) * w * np.cos(w * t_rel_grid)
                # find sign changes in derivative after 'now'
                idx_start = int(max(0, np.searchsorted(grid_epochs, now_ts) - 1))
                for k in range(idx_start, len(grid_epochs) - 1):
                    d0 = deriv_grid[k]
                    d1 = deriv_grid[k + 1]
                    t0 = float(grid_epochs[k])
                    t1 = float(grid_epochs[k + 1])
                    if abs(d0) < EPS_DERIV and t0 >= now_ts:
                        candidates.append(t0)
                    if d0 * d1 < 0:
                        # refine root analytically between t0 and t1
                        def _deriv_local(x):
                            t_rel_local = x - t_anchor
                            s = 0.0
                            for jj, cc in enumerate(self._constituents):
                                ww = omegas[cc]
                                s += -float(A[jj]) * ww * math.sin(ww * t_rel_local) + float(B[jj]) * ww * math.cos(ww * t_rel_local)
                            return s

                        root = None
                        try:
                            root = (lambda f, a, b: self._find_root_bisect_helper(f, a, b))( _deriv_local, t0, t1)
                        except Exception:
                            root = None
                        if root is not None and root >= now_ts:
                            candidates.append(root)
            # fallback: if no grid or no candidates, sample a small forward window analytically to find extrema
            if not candidates:
                # sample analytic derivative across next 48 hours at GRID_SECONDS_DEFAULT spacing
                span_end = now_ts + 48 * 3600
                pts = np.arange(now_ts, span_end + 1, GRID_SECONDS_DEFAULT)
                deriv_vals = []
                for x in pts:
                    deriv_vals.append(self._derivative_analytic(x, omegas, A, B, t_anchor))
                deriv_vals = np.array(deriv_vals)
                for k in range(len(pts) - 1):
                    if deriv_vals[k] * deriv_vals[k + 1] < 0:
                        candidates.append(float(pts[k]))

            # classify candidates into maxima/minima using second derivative evaluation
            maxima: List[Tuple[float, float]] = []
            minima: List[Tuple[float, float]] = []
            for rt in candidates:
                h = self._height_analytic(rt, omegas, A, B, t_anchor)
                sec = self._second_derivative_analytic(rt, omegas, A, B, t_anchor)
                if sec < -EPS_DERIV:
                    maxima.append((rt, h))
                elif sec > EPS_DERIV:
                    minima.append((rt, h))

            next_high_tuple: Optional[Tuple[float, float]] = None
            next_low_tuple: Optional[Tuple[float, float]] = None
            if maxima:
                next_high_tuple = min(maxima, key=lambda x: x[0])
            if minima:
                next_low_tuple = min(minima, key=lambda x: x[0])

            # fallback: use grid-sampled argmax/argmin
            if next_high_tuple is None and grid_pred.size:
                idx = int(np.argmax(grid_pred[np.searchsorted(grid_epochs, now_ts):]))
                idx_abs = np.searchsorted(grid_epochs, now_ts) + idx
                if idx_abs < len(grid_epochs):
                    next_high_tuple = (float(grid_epochs[idx_abs]), float(grid_pred[idx_abs]))
            if next_low_tuple is None and grid_pred.size:
                idx = int(np.argmin(grid_pred[np.searchsorted(grid_epochs, now_ts):]))
                idx_abs = np.searchsorted(grid_epochs, now_ts) + idx
                if idx_abs < len(grid_epochs):
                    next_low_tuple = (float(grid_epochs[idx_abs]), float(grid_pred[idx_abs]))

            if next_high_tuple is not None:
                nh_ts, nh_h = next_high_tuple
                # refine using analytic bisection around a small window
                try:
                    root = self._refine_extremum_around(nh_ts - GRID_SECONDS_DEFAULT, nh_ts + GRID_SECONDS_DEFAULT, omegas, A, B, t_anchor)
                    if root is not None:
                        nh_ts = root
                        nh_h = self._height_analytic(root, omegas, A, B, t_anchor)
                except Exception:
                    pass
                # ensure vertical datum offset included (analytic path includes it already)
                if getattr(self, "VERTICAL_DATUM_OFFSET_M", 0.0) and not (grid_pred.size and nh_ts >= grid_epochs[0] and nh_ts <= grid_epochs[-1]):
                    nh_h = nh_h + float(self.VERTICAL_DATUM_OFFSET_M)
                next_high = datetime.fromtimestamp(nh_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                next_high_height = float(round(float(nh_h), 3))
            if next_low_tuple is not None:
                nl_ts, nl_h = next_low_tuple
                try:
                    root = self._refine_extremum_around(nl_ts - GRID_SECONDS_DEFAULT, nl_ts + GRID_SECONDS_DEFAULT, omegas, A, B, t_anchor)
                    if root is not None:
                        nl_ts = root
                        nl_h = self._height_analytic(root, omegas, A, B, t_anchor)
                except Exception:
                    pass
                if getattr(self, "VERTICAL_DATUM_OFFSET_M", 0.0) and not (grid_pred.size and nl_ts >= grid_epochs[0] and nl_ts <= grid_epochs[-1]):
                    nl_h = nl_h + float(self.VERTICAL_DATUM_OFFSET_M)
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
                try:
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
                except Exception:
                    tide_strength_value = float(_compute_tide_strength_phase_heuristic(_coerce_phase(moon_phases[0] if moon_phases else None)))
            else:
                tide_strength_value = float(_compute_tide_strength_phase_heuristic(_coerce_phase(moon_phases[0] if moon_phases else None)))
        except Exception:
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
                "t_ref": float(T_REF_EPOCH),
                "period_seconds": float(period_seconds),
                "coef_vec_len": int(self._coef_vec.size),
                "vertical_datum_offset_m": float(getattr(self, "VERTICAL_DATUM_OFFSET_M", 0.0)),
                "utide_applied": True,
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

    # --- analytic helpers used by the grid-based peak finder ---
    def _height_analytic(self, epoch_ts: float, omegas: Dict[str, float], A: np.ndarray, B: np.ndarray, t_anchor: float) -> float:
        t_rel = epoch_ts - t_anchor
        s = float(self._bias) if hasattr(self, "_bias") else 0.0
        for j, c in enumerate(self._constituents):
            w = omegas[c]
            s += float(A[j]) * math.cos(w * t_rel) + float(B[j]) * math.sin(w * t_rel)
        if getattr(self, "VERTICAL_DATUM_OFFSET_M", 0.0):
            s = s + float(self.VERTICAL_DATUM_OFFSET_M)
        return s

    def _derivative_analytic(self, epoch_ts: float, omegas: Dict[str, float], A: np.ndarray, B: np.ndarray, t_anchor: float) -> float:
        t_rel = epoch_ts - t_anchor
        s = 0.0
        for j, c in enumerate(self._constituents):
            w = omegas[c]
            s += -float(A[j]) * w * math.sin(w * t_rel) + float(B[j]) * w * math.cos(w * t_rel)
        return s

    def _second_derivative_analytic(self, epoch_ts: float, omegas: Dict[str, float], A: np.ndarray, B: np.ndarray, t_anchor: float) -> float:
        t_rel = epoch_ts - t_anchor
        s = 0.0
        for j, c in enumerate(self._constituents):
            w = omegas[c]
            s += -float(A[j]) * (w ** 2) * math.cos(w * t_rel) - float(B[j]) * (w ** 2) * math.sin(w * t_rel)
        return s

    def _find_root_bisect_helper(self, f, a: float, b: float, maxiter: int = 60, tol: float = BISECT_TOL_SEC) -> Optional[float]:
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

    def _refine_extremum_around(self, a: float, b: float, omegas: Dict[str, float], A: np.ndarray, B: np.ndarray, t_anchor: float) -> Optional[float]:
        """
        Refine a candidate extremum by finding a root of derivative between a and b.
        """
        try:
            def deriv_local(x):
                return self._derivative_analytic(x, omegas, A, B, t_anchor)

            root = self._find_root_bisect_helper(deriv_local, a, b)
            return root
        except Exception:
            return None