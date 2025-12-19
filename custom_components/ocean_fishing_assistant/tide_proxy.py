# custom_components/ocean_fishing_assistant/tide_proxy.py
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

# timezone conversion
from zoneinfo import ZoneInfo

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# numeric tolerances
EPS_DERIV = 1e-10
EPS_ROOT = 1e-9
BISECT_TOL_SEC = 1e-1  # stopping tolerance for root bisection (seconds)
GRID_SECONDS_DEFAULT = 60  # 1 minute grid for better extrema resolution

CONSTITUENT_PERIOD_HOURS: Dict[str, float] = {
    "M2": 12.4206,
    "S2": 12.0,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
    "P1": 24.0659,
    "Q1": 26.8683,
    "S1": 24.0,
    "M4": 12.4206 / 2.0,
    "M6": 12.4206 / 3.0,
}

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


def _to_epoch_seconds(ts: Any) -> int:
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if isinstance(ts, float):
        return int(ts)
    if isinstance(ts, datetime):
        return int(ts.astimezone(timezone.utc).timestamp())
    if isinstance(ts, str):
        s = ts.strip()
        try:
            v = float(s)
            if v > 1e12:
                v = v / 1000.0
            return int(v)
        except Exception:
            pass
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


def _compute_tide_strength_phase_heuristic(phase: Optional[float]) -> float:
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


# UTide is a strict requirement per your request
try:
    import utide  # type: ignore
    from utide import harmonics  # type: ignore
    # utide.reconstruct is the canonical reconstruction module; import it for use.
    from utide import reconstruct  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "UTide is required for tide reconstruction but is not available. "
        "Install UTide in the Home Assistant environment so this integration can run."
    ) from exc


def nfactors(jd: float, names: Sequence[str], latitude: float = 0.0) -> Dict[str, Dict[str, float]]:
    # Uses utide.harmonics.FUV (same behavior as before)
    constit_index = getattr(utide, "constit_index_dict", None)
    if constit_index is None:
        raise RuntimeError("utide.constit_index_dict not found in installed UTide; cannot map constituent names.")

    lind_list: List[int] = []
    name_idx_map: Dict[str, int] = {}
    for nm in names:
        if nm not in constit_index:
            raise KeyError(f"Constituent '{nm}' not known to utide.constit_index_dict.")
        idx = int(constit_index[nm]) - 1
        if idx < 0:
            raise KeyError(f"Constituent '{nm}' maps to invalid index {idx+1} in utide.constit_index_dict.")
        lind_list.append(int(idx))
        name_idx_map[nm] = int(idx)
        _LOGGER.debug("nfactors: mapped %s -> index %d", nm, idx + 1)

    if not lind_list:
        raise RuntimeError("No constituents provided to nfactors.")

    lind = np.atleast_1d(lind_list).astype(int)
    t_arr = np.atleast_1d(jd)
    ngflgs = [False, False, False, False]
    F, U, V = harmonics.FUV(t_arr, jd, lind, float(latitude), ngflgs)

    F_row = np.asarray(F).reshape((1, -1))[0]
    U_row = np.asarray(U).reshape((1, -1))[0]

    out: Dict[str, Dict[str, float]] = {}
    for nm, idx in name_idx_map.items():
        pos = lind.tolist().index(int(idx))
        fval = float(F_row[pos])
        u_raw = float(U_row[pos])
        if abs(u_raw) <= 1.1:
            uval_deg = u_raw * 360.0
        elif abs(u_raw) <= 2.0 * math.pi + 0.1:
            uval_deg = math.degrees(u_raw)
        else:
            uval_deg = u_raw
        _LOGGER.debug("nfactors: %s -> f=%s u_raw=%s u_deg=%s", nm, fval, u_raw, uval_deg)
        out[nm] = {"f": fval, "u": uval_deg}

    return out


class TideProxy:
    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        ttl: int = _DEFAULT_TTL,
        *,
        coef_vec: Optional[Sequence[float]] = None,
        default_m2_amp: float = 0.3,
        bias: float = 0.0,
        auto_clamp_enabled: bool = False,
        min_height_floor: Optional[float] = None,
        max_amplitude_m: Optional[float] = None,
        phase_offset_hours: float = 0.0,
    ):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

        self._constituents = list(CONSTITUENT_PERIOD_HOURS.keys())
        self._bias = float(bias)
        self._auto_clamp_enabled = bool(auto_clamp_enabled)
        self._min_height_floor = None if min_height_floor is None else float(min_height_floor)
        self._max_amplitude_m = None if max_amplitude_m is None else float(max_amplitude_m)
        self._phase_offset_hours = float(phase_offset_hours)

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

        if coef_vec is not None:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size == 2 * len(self._constituents):
                self._coef_vec = arr.copy()
            else:
                _LOGGER.warning("coef_vec length mismatch; using default built-ins")
                self._coef_vec = self._build_default_coef_vec(default_m2_amp)
                _LOGGER.info("Using default zero-phase coefficients (built-in) with default_m2_amp=%.3f m", default_m2_amp)
        else:
            # Use zero-phase default coefficients so t_anchor controls timing predictably.
            self._coef_vec = self._build_default_coef_vec(default_m2_amp)
            _LOGGER.info("Using default zero-phase coefficients (built-in) with default_m2_amp=%.3f m", default_m2_amp)

        try:
            mean_abs_A = float(np.mean(np.abs(self._coef_vec[0::2])))
            if mean_abs_A > 10.0:
                _LOGGER.warning(
                    "Large mean coefficient amplitude detected (%.3f). Ensure coef_vec elements are meters (A_cos,B_sin) not cm.",
                    mean_abs_A,
                )
        except Exception:
            pass

        _LOGGER.debug(
            "TideProxy initialized lat=%s lon=%s coef_len=%d bias=%.6f clamp=%s phase_offset_hours=%.3f",
            self.latitude,
            self.longitude,
            self._coef_vec.size,
            self._bias,
            self._auto_clamp_enabled,
            self._phase_offset_hours,
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

    async def get_tide_for_timestamps(self, timestamps: Sequence[Any], *, location_tz: str) -> Dict[str, Any]:
        """
        Compute tide metadata aligned to the provided timestamps.

        location_tz: required IANA timezone name (string). Strict: raise if missing/invalid.
        """
        if not location_tz:
            raise ValueError("location_tz is required (strict)")

        # We keep the existing UTC-based tide model outputs but require the tz param for callers.
        now = dt_util.now().astimezone(timezone.utc)

        if not timestamps:
            return {
                "timestamps": [],
                "tide_phase": [],
                "moon_phase": [],
                "tide_strength": 0.0,
                "confidence": "no_timestamps",
                "source": "tide_proxy",
                "next_high": None,
                "next_low": None,
            }

        try:
            new_keys = _normalize_timestamps_for_cache(timestamps)
        except Exception:
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
                        "tide_phase": [None] * len(timestamps),
                        "confidence": "bad_timestamps",
                        "source": "tide_proxy",
                    }
                dt_objs_try.append(parsed.astimezone(timezone.utc))
            new_keys = [int(dt.timestamp()) for dt in dt_objs_try]

        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            cached_keys = self._cache.get("timestamps")
            if cached_keys is not None and cached_keys == new_keys:
                return self._cache["raw_tide"]

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
                return {
                    "timestamps": [str(t) for t in timestamps],
                    "tide_phase": [None] * len(timestamps),
                    "confidence": "bad_timestamps",
                    "source": "tide_proxy",
                }
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)

        try:
            await self._ensure_loaded()
        except Exception:
            _LOGGER.exception("Skyfield unavailable")
            return {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_phase": [None] * len(dt_objs),
                "confidence": "astronomical_unavailable",
                "source": "astronomical_unavailable",
            }

        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            moon_transit_dt = None

        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR

        # Do NOT apply any longitude-derived phase shift here.
        t_anchor = anchor_epoch
        if moon_transit_dt is None:
            _LOGGER.debug("Moon transit not found; using anchor_epoch without longitude-derived phase shift")
        else:
            _LOGGER.debug("Moon transit found: transit=%s anchor_epoch=%s (UTC) phase_offset_hours=%.3f",
                          moon_transit_dt.isoformat().replace("+00:00", "Z") if moon_transit_dt else "None",
                          anchor_epoch,
                          self._phase_offset_hours)

        # Apply manual phase offset (hours) for empirical/site tuning (positive -> shift anchor forward)
        try:
            if float(self._phase_offset_hours) != 0.0:
                t_anchor = float(t_anchor) + float(self._phase_offset_hours) * _SECONDS_PER_HOUR
                _LOGGER.debug("Applied manual phase_offset_hours: %.3f -> new t_anchor=%s", self._phase_offset_hours, datetime.fromtimestamp(t_anchor, tz=timezone.utc).isoformat().replace("+00:00", "Z"))
        except Exception:
            pass

        periods_sec = {k: (CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in self._constituents}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        coef_arr = np.asarray(self._coef_vec, dtype=float)
        A = coef_arr[0::2].astype(float)
        B = coef_arr[1::2].astype(float)

        A_orig = A.copy()
        B_orig = B.copy()

        jd_anchor = float(t_anchor) / 86400.0 + 2440587.5
        _LOGGER.debug("Nodal correction context: t_anchor=%s jd_anchor=%s phase_offset_hours=%.3f", t_anchor, jd_anchor, self._phase_offset_hours)

        # Compute nodal factors (UTide harmonics.FUV) for the anchor JD
        try:
            nf = await self.hass.async_add_executor_job(nfactors, jd_anchor, self._constituents, float(self.latitude))
        except Exception:
            _LOGGER.exception("Failed to compute nodal factors via UTide")
            raise

        # Convert internal coef_vec (A_cos,B_sin) into amplitude+phase (meters, degrees)
        # UTide convention: amplitude (positive scalar) and phase in degrees (we will apply nodal phase u in degrees)
        amp_list: List[float] = []
        phase_deg_list: List[float] = []
        for i, cname in enumerate(self._constituents):
            Ai = float(A[i])
            Bi = float(B[i])
            R = math.hypot(Ai, Bi)
            # phase from A (cos) and B (sin): phi = atan2(B, A) in radians; convert to degrees
            phi_rad = math.atan2(Bi, Ai)
            phi_deg = math.degrees(phi_rad)
            amp_list.append(float(R))
            phase_deg_list.append(float(phi_deg))

        # Apply nodal corrections (f scales amplitude, u adds to phase)
        for i, cname in enumerate(self._constituents):
            if cname not in nf:
                raise KeyError(f"nfactors did not return entry for constituent '{cname}'")
            fval = float(nf[cname].get("f", 1.0))
            udeg = float(nf[cname].get("u", 0.0))
            amp_list[i] = float(amp_list[i] * fval)
            phase_deg_list[i] = float(phase_deg_list[i] + udeg)

        # Prepare times for UTide reconstruct: UTide expects times in decimal days (Julian day or days since epoch depending on API).
        # UTide python reconstruct() accepts times in days (e.g. Julian day numbers). We'll pass JD relative to epoch used above: use JD (UTC)
        jd_times = np.array([dt.timestamp() / 86400.0 + 2440587.5 for dt in dt_objs], dtype=float)

        # Prepare UTide input structure per reconstruct API
        # Build a dictionary mapping UTide expected names -> arrays
        try:
            # Log debug summary before calling UTide
            _LOGGER.debug(
                "UTide reconstruction inputs: n_times=%d n_const=%d jd_anchor=%s sample_jd0=%s",
                len(jd_times),
                len(self._constituents),
                jd_anchor,
                jd_times[0] if jd_times.size > 0 else "none",
            )
            _LOGGER.debug("UTide constituents=%s", self._constituents)
            _LOGGER.debug("UTide amplitudes(sample)=%s", amp_list[:6])
            _LOGGER.debug("UTide phases_deg(sample)=%s", phase_deg_list[:6])
        except Exception:
            pass

        # Call UTide reconstruct inside executor (blocking)
        def _blocking_utide_reconstruct(jd_arr, names, amps, phases_deg):
            """
            Use utide.reconstruct to compute reconstructed tide heights for JD array.
            Note: UTide reconstruct API historically accepts arrays and a 'cons' or similar structure.
            We call reconstruct.reconstruct(...) but signature may vary across UTide versions.
            The typical pattern:
                rec = reconstruct.reconstruct(t, amp, pha, con=None, names=names)
            If your utide version requires a different signature, adjust this helper accordingly.
            """
            # Convert to numpy arrays of correct dtype
            t_arr = np.asarray(jd_arr, dtype=float)
            amp_arr = np.asarray(amps, dtype=float)
            pha_arr = np.asarray(phases_deg, dtype=float)

            # The utide.reconstruct module provides a reconstruct function; try a couple of common signatures.
            try:
                # Signature attempt 1 (common): reconstruct.reconstruct(t, amp, pha, cons=None, names=names)
                try:
                    return reconstruct.reconstruct(t_arr, amp_arr, pha_arr, names=names)
                except TypeError:
                    # Signature attempt 2: reconstruct.reconstruct(t, names, amp, pha)
                    try:
                        return reconstruct.reconstruct(t_arr, names, amp_arr, pha_arr)
                    except TypeError:
                        # Signature attempt 3: reconstruct.reconstruct(t, amp, pha, con=None)
                        return reconstruct.reconstruct(t_arr, amp_arr, pha_arr)
            except Exception as exc:
                # Re-raise with context to the outer caller
                raise RuntimeError(f"UTide reconstruct call failed: {exc}") from exc

        try:
            rec_result = await self.hass.async_add_executor_job(
                _blocking_utide_reconstruct,
                jd_times,
                self._constituents,
                amp_list,
                phase_deg_list,
            )
        except Exception:
            _LOGGER.exception("UTide reconstruction failed")
            raise

        # UTide reconstruct return types vary across versions; try to normalize:
        # - Some UTide wrappers return a numpy array of reconstructed heights.
        # - Some return a dict-like object. We handle both.
        if isinstance(rec_result, np.ndarray):
            rec_heights = np.asarray(rec_result, dtype=float)
        elif isinstance(rec_result, (list, tuple)):
            rec_heights = np.asarray(rec_result, dtype=float)
        elif isinstance(rec_result, dict):
            # Common dict key is 'recon' or 'y' or 'h'; try to pick a sensible field
            if "recon" in rec_result:
                rec_heights = np.asarray(rec_result["recon"], dtype=float)
            elif "y" in rec_result:
                rec_heights = np.asarray(rec_result["y"], dtype=float)
            elif "h" in rec_result:
                rec_heights = np.asarray(rec_result["h"], dtype=float)
            else:
                # last resort: try to locate first array-like value
                found = None
                for v in rec_result.values():
                    if isinstance(v, (list, tuple, np.ndarray)):
                        found = v
                        break
                if found is None:
                    raise RuntimeError("UTide reconstruct returned unexpected dict shape; cannot locate reconstructed heights")
                rec_heights = np.asarray(found, dtype=float)
        else:
            raise RuntimeError("UTide reconstruct returned unexpected result type; cannot normalize reconstructed heights")

        # Sanity check: rec_heights length must equal input times length
        if rec_heights.size != len(dt_objs):
            raise RuntimeError("UTide reconstructed heights length mismatch with requested timestamps")

        # Optional clamping / scaling (same semantics as previous code)
        if self._auto_clamp_enabled:
            try:
                if self._max_amplitude_m is not None:
                    current_half = (float(np.max(rec_heights)) - float(np.min(rec_heights))) / 2.0
                    if current_half > float(self._max_amplitude_m) and current_half > 1e-12:
                        mean_val = float(np.mean(rec_heights))
                        scale = float(self._max_amplitude_m) / current_half
                        rec_heights = mean_val + (rec_heights - mean_val) * scale
                if self._min_height_floor is not None:
                    rec_heights = np.maximum(rec_heights, float(self._min_height_floor))
            except Exception:
                _LOGGER.exception("Error applying clamp/scale to UTide reconstruction")

        # Now use reconstructed heights to find extrema and classify tide phase as before.
        next_high: Optional[str] = None
        next_low: Optional[str] = None

        try:
            now_ts = now.timestamp()
            t_epochs = np.array([dt.timestamp() for dt in dt_objs], dtype=float)
            order = np.argsort(t_epochs)
            t_sorted = t_epochs[order]
            pred_sorted = np.asarray(rec_heights, dtype=float)[order]

            n_sorted = len(t_sorted)
            first_future_idx_sorted = n_sorted
            for i_s, tval in enumerate(t_sorted):
                if tval >= now_ts:
                    first_future_idx_sorted = i_s
                    break

            def _height_at_epoch_from_utide(epoch_ts: float) -> float:
                # Interpolate linearly between provided times (safe since input grid is hourly)
                if epoch_ts <= t_sorted[0]:
                    return float(pred_sorted[0])
                if epoch_ts >= t_sorted[-1]:
                    return float(pred_sorted[-1])
                idx = np.searchsorted(t_sorted, epoch_ts) - 1
                idx = max(0, min(idx, len(t_sorted) - 2))
                t0 = t_sorted[idx]
                t1 = t_sorted[idx + 1]
                y0 = float(pred_sorted[idx])
                y1 = float(pred_sorted[idx + 1])
                return y0 + (y1 - y0) * ((epoch_ts - t0) / (t1 - t0)) if (t1 - t0) != 0 else y0

            def _derivative_at_epoch_numeric(epoch_ts: float) -> float:
                # Numeric derivative via central difference using small step (seconds)
                h = 60.0  # 1 minute step for derivative estimation
                y_plus = _height_at_epoch_from_utide(epoch_ts + h)
                y_minus = _height_at_epoch_from_utide(epoch_ts - h)
                return (y_plus - y_minus) / (2.0 * h)

            def _second_derivative_at_epoch_numeric(epoch_ts: float) -> float:
                h = 60.0
                y_plus = _height_at_epoch_from_utide(epoch_ts + h)
                y = _height_at_epoch_from_utide(epoch_ts)
                y_minus = _height_at_epoch_from_utide(epoch_ts - h)
                return (y_plus - 2.0 * y + y_minus) / (h * h)

            # Build a coarse grid and search for derivative sign changes similar to previous algorithm
            candidates: List[float] = []
            scan_start = max(now_ts, float(t_sorted[0])) if t_sorted.size > 0 else now_ts
            scan_end = float(t_sorted[-1]) if t_sorted.size > 0 else now_ts

            if scan_end > scan_start:
                step = float(GRID_SECONDS_DEFAULT)
                grid = np.arange(scan_start, scan_end + 0.5 * step, step, dtype=float)
                d_grid = np.zeros_like(grid)
                for gi, gt in enumerate(grid):
                    try:
                        d_grid[gi] = float(_derivative_at_epoch_numeric(gt))
                    except Exception:
                        d_grid[gi] = 0.0

                near_zero_idx = np.where(np.abs(d_grid) < EPS_DERIV)[0]
                for idx in near_zero_idx:
                    t_candidate = float(grid[idx])
                    if t_candidate >= now_ts:
                        candidates.append(t_candidate)

                prod = d_grid[:-1] * d_grid[1:]
                sign_change_idx = np.where(prod < 0)[0]
                for idx in sign_change_idx:
                    a = float(grid[idx])
                    b = float(grid[idx + 1])

                    def _f_deriv(x):
                        return float(_derivative_at_epoch_numeric(x))

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

                    try:
                        root = _find_root_bisection(_f_deriv, a, b)
                        if root is not None and root >= now_ts:
                            candidates.append(root)
                    except Exception:
                        continue

            candidates = sorted(set([float(x) for x in candidates]))

            maxima: List[Tuple[float, float]] = []
            minima: List[Tuple[float, float]] = []

            delta_sec = max(1.0, float(GRID_SECONDS_DEFAULT) / 8.0)
            classification_debug: List[Tuple[str, float, float, float, str]] = []

            for rt in candidates:
                try:
                    t_before = rt - delta_sec
                    t_after = rt + delta_sec
                    t_before = max(t_before, scan_start)
                    t_after = min(t_after, scan_end)

                    d_before = _derivative_at_epoch_numeric(t_before)
                    d_after = _derivative_at_epoch_numeric(t_after)

                    cls = ""
                    sec = None
                    if d_before > 0.0 and d_after < 0.0:
                        maxima.append((rt, _height_at_epoch_from_utide(rt)))
                        cls = "max_sign"
                    elif d_before < 0.0 and d_after > 0.0:
                        minima.append((rt, _height_at_epoch_from_utide(rt)))
                        cls = "min_sign"
                    else:
                        sec = _second_derivative_at_epoch_numeric(rt)
                        if sec < -EPS_DERIV:
                            maxima.append((rt, _height_at_epoch_from_utide(rt)))
                            cls = "max_sec"
                        elif sec > EPS_DERIV:
                            minima.append((rt, _height_at_epoch_from_utide(rt)))
                            cls = "min_sec"
                        else:
                            cls = "ambiguous"

                    if len(classification_debug) < 12:
                        sec_val = sec if sec is not None else _second_derivative_at_epoch_numeric(rt)
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
                    _LOGGER.debug("candidate classification failed at %s: %s", rt, e, exc_info=True)
                    continue

            try:
                _LOGGER.debug(
                    "next-extrema classify: scan_start=%s scan_end=%s grid_step=%ss classification_debug(sample)=%s",
                    datetime.fromtimestamp(scan_start, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    datetime.fromtimestamp(scan_end, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    GRID_SECONDS_DEFAULT,
                    classification_debug,
                )
            except Exception:
                pass

            next_high_tuple: Optional[Tuple[float, float]] = None
            next_low_tuple: Optional[Tuple[float, float]] = None
            if maxima:
                next_high_tuple = min(maxima, key=lambda x: x[0])
            if minima:
                next_low_tuple = min(minima, key=lambda x: x[0])

            if next_high_tuple is None or next_low_tuple is None:
                if first_future_idx_sorted < t_sorted.size:
                    a_idx = max(0, first_future_idx_sorted - 1)
                    b_idx = min(n_sorted - 1, first_future_idx_sorted + 1)
                    a = float(t_sorted[a_idx])
                    b = float(t_sorted[b_idx])
                    if b > a:
                        root = None
                        try:
                            root = (a + b) / 2.0
                        except Exception:
                            root = None
                        if root is not None and root >= now_ts:
                            h = _height_at_epoch_from_utide(root)
                            sec = _second_derivative_at_epoch_numeric(root)
                            if sec < -EPS_DERIV and next_high_tuple is None:
                                next_high_tuple = (root, h)
                            elif sec > EPS_DERIV and next_low_tuple is None:
                                next_low_tuple = (root, h)

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
                nh_ts, _ = next_high_tuple
                next_high = datetime.fromtimestamp(nh_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            if next_low_tuple is not None:
                nl_ts, _ = next_low_tuple
                next_low = datetime.fromtimestamp(nl_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")

            # Add helpful debug about selected extrema
            try:
                _LOGGER.debug(
                    "Selected next extrema: next_high=%s next_low=%s (first_future_idx_sorted=%s scan_range=%s->%s)",
                    next_high,
                    next_low,
                    first_future_idx_sorted,
                    datetime.fromtimestamp(scan_start, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    datetime.fromtimestamp(scan_end, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                )
            except Exception:
                pass
        except Exception:
            _LOGGER.debug("Failed to compute next_high/next_low", exc_info=True)

        # Compute moon phases for the timestamps (unchanged, still using Skyfield)
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

        # Tide strength estimate (unchanged)
        tide_strength_value = 0.5
        try:
            if sf_ts is not None and sf_eph is not None and len(dt_objs) > 0:
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
                d_ref = 0.00257
                raw = 0.5 * (1.0 + math.cos(delta)) * (d_ref / max(d_moon_au, 1e-9)) ** 3
                d_min_plausible = 0.0024
                max_raw = 0.5 * (1.0 + 1.0) * (d_ref / d_min_plausible) ** 3
                strength = raw / max_raw if max_raw > 0 else raw
                tide_strength_value = float(max(0.0, min(1.0, strength)))
            else:
                tide_strength_value = float(_compute_tide_strength_phase_heuristic(_coerce_phase(moon_phases[0] if moon_phases else None)))
        except Exception as e:
            _LOGGER.debug("Physical tide_strength proxy failed, falling back to phase heuristic: %s", e, exc_info=True)
            tide_strength_value = float(_compute_tide_strength_phase_heuristic(_coerce_phase(moon_phases[0] if moon_phases else None)))

        # --- NEW: compute tide_phase as strings (rising/falling/high/low) ---
        try:
            maxima_epochs = set([float(x[0]) for x in (maxima if 'maxima' in locals() else [])])
            minima_epochs = set([float(x[0]) for x in (minima if 'minima' in locals() else [])])

            tide_phase_list: List[str] = []
            def _derivative_at_epoch_local(epoch_ts: float) -> float:
                # numeric derivative based on UTide reconstructed heights
                # reuse interpolation and numeric derivative helpers from above
                return float(np.sign(_derivative_at_epoch_numeric(epoch_ts))) * abs(_derivative_at_epoch_numeric(epoch_ts))

            for dt in dt_objs:
                epoch = float(dt.timestamp())
                marked = False
                for me in maxima_epochs:
                    if abs(epoch - me) <= 60.0:
                        tide_phase_list.append("high")
                        marked = True
                        break
                if marked:
                    continue
                for me in minima_epochs:
                    if abs(epoch - me) <= 60.0:
                        tide_phase_list.append("low")
                        marked = True
                        break
                if marked:
                    continue
                dval = _derivative_at_epoch_numeric(epoch)
                if not np.isfinite(dval):
                    raise RuntimeError(f"Non-finite derivative while classifying tide_phase at epoch {epoch}")
                if abs(dval) < 1e-12:
                    # treat near-zero derivative as ambiguous and fail strictly
                    raise RuntimeError(f"Zero/ambiguous derivative while classifying tide_phase at epoch {epoch}")
                if dval > 0.0:
                    tide_phase_list.append("rising")
                else:
                    tide_phase_list.append("falling")

            if len(tide_phase_list) != len(dt_objs):
                raise RuntimeError("Internal tide_phase classification length mismatch")
        except Exception as exc:
            _LOGGER.exception("Failed to compute tide_phase strings: %s", exc)
            raise

        PHASE_NAME_MAP = {
            "rising": "Rising",
            "falling": "Falling",
            "high": "High Tide",
            "low": "Low Tide",
        }
        tide_phase_name_list: List[str] = []
        try:
            for p in tide_phase_list:
                if not isinstance(p, str):
                    raise RuntimeError(f"Invalid tide_phase value (not a string): {p!r}")
                key = p.lower()
                if key not in PHASE_NAME_MAP:
                    raise RuntimeError(f"Unexpected tide_phase value: {p!r}")
                tide_phase_name_list.append(PHASE_NAME_MAP[key])
            if len(tide_phase_name_list) != len(tide_phase_list):
                raise RuntimeError("tide_phase_name length mismatch")
        except Exception as exc:
            _LOGGER.exception("Failed to map tide_phase -> tide_phase_name: %s", exc)
            raise

        next_high_obj = None
        next_low_obj = None
        if next_high is not None:
            try:
                next_high_obj = {"timestamp": next_high}
            except Exception:
                next_high_obj = None
        if next_low is not None:
            try:
                next_low_obj = {"timestamp": next_low}
            except Exception:
                next_low_obj = None

        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            # Removed tide_height_m per request
            # Add explicit numeric moon_phase separate from tide_phase strings
            "moon_phase": moon_phases,
            "tide_phase": tide_phase_list,
            "tide_phase_name": tide_phase_name_list,
            "tide_strength": float(round(tide_strength_value, 3)),
            "confidence": "in_memory_model",
            "source": "utide_reconstruction",
            "_helpers": {
                "constituents": self._constituents,
                "t_anchor": float(t_anchor),
                "jd_anchor": float(jd_anchor),
                "period_seconds": float(period_seconds),
                "coef_vec_len": int(self._coef_vec.size),
                "phase_offset_hours": float(self._phase_offset_hours),
            },
            "next_high": next_high_obj,
            "next_low": next_low_obj,
        }

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
        *,
        location_tz: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute indices mapping for periods.

        location_tz: required IANA timezone name (string). Strict: raise if missing/invalid.
        Returns a dict keyed by local-date (YYYY-MM-DD) with period entries containing 'indices', 'start', 'end' (UTC ISO strings).
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
            tzinfo_local = ZoneInfo(location_tz)
        except Exception as exc:
            raise ValueError(f"Invalid location_tz '{location_tz}': {exc}") from exc

        local_dt_objs: List[datetime] = [dt.astimezone(tzinfo_local) for dt in dt_objs]
        index_dt_pairs = list(enumerate(dt_objs))  # dt_objs are UTC (for index matching)
        dates_needed = sorted({local_dt.date() for local_dt in local_dt_objs})

        await self._ensure_loaded()
        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac
        earth = sf_eph["earth"] if sf_eph is not None else None
        topos = sf_wgs.latlon(self.latitude, self.longitude) if sf_wgs is not None else None

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
                    # Expand search window (local) for sunrise/sunset to ensure we capture events near midnight boundaries
                    expand = timedelta(hours=12)
                    search_start_local = local_day_start - expand
                    search_end_local = local_day_end + expand

                    # convert local search bounds to UTC and to skyfield timescale
                    search_start_utc = search_start_local.astimezone(timezone.utc)
                    search_end_utc = search_end_local.astimezone(timezone.utc)
                    t0_exp = sf_ts.utc(search_start_utc.year, search_start_utc.month, search_start_utc.day, search_start_utc.hour, search_start_utc.minute, search_start_utc.second)
                    t1_exp = sf_ts.utc(search_end_utc.year, search_end_utc.month, search_end_utc.day, search_end_utc.hour, search_end_utc.minute, search_end_utc.second)

                    f = sf_almanac.sunrise_sunset(sf_eph, topos)
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
                        # convert to local tz for selection
                        evt_dt_local = evt_dt_utc.astimezone(tzinfo_local)
                        if bool(ev):
                            sunrise_candidates.append(evt_dt_local)
                        else:
                            sunset_candidates.append(evt_dt_local)
                    if not sunrise_candidates or not sunset_candidates:
                        raise RuntimeError(f"Unable to determine sunrise or sunset for local date {d.isoformat()} at lat={self.latitude},lon={self.longitude}")

                    # choose sunrise closest to morning_target (local 06:00) and sunset closest to evening_target (local 18:00)
                    morning_target = local_day_start + timedelta(hours=6)
                    evening_target = local_day_start + timedelta(hours=18)
                    sunrise_dt_local = min(sunrise_candidates, key=lambda e: abs((e - morning_target).total_seconds()))
                    sunset_dt_local = min(sunset_candidates, key=lambda e: abs((e - evening_target).total_seconds()))

                    dawn_start_local = sunrise_dt_local - timedelta(hours=dawn_window_hours)
                    dawn_end_local = sunrise_dt_local + timedelta(hours=dawn_window_hours)
                    dusk_start_local = sunset_dt_local - timedelta(hours=dawn_window_hours)
                    dusk_end_local = sunset_dt_local + timedelta(hours=dawn_window_hours)

                    # convert local windows to UTC for index matching
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