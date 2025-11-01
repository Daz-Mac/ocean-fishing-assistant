"""
Thin wrapper to validate and canonicalize data immediately after fetch.
This keeps the fetcher focused on retrieval and ensures downstream modules
receive a stable SI-structured payload.
"""
from typing import Dict, Any
from .ocean_scoring import compute_score, MissingDataError

class DataFormatter:
    def _convert_imperial_to_metric(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert common imperial keys into canonical metric keys in-place on a shallow copy."""
        out = dict(data)
        # temperatures
        if "temperature_c" not in out:
            if "temperature_f" in out:
                tf = out.get("temperature_f")
                if isinstance(tf, (list, tuple)):
                    out["temperature_c"] = [round((float(v) - 32.0) * 5.0 / 9.0, 2) if v is not None else None for v in tf]
                else:
                    try:
                        out["temperature_c"] = float((float(tf) - 32.0) * 5.0 / 9.0)
                    except Exception:
                        pass
        # wind
        if "wind_m_s" not in out:
            if "wind_mph" in out:
                wm = out.get("wind_mph")
                if isinstance(wm, (list, tuple)):
                    out["wind_m_s"] = [round(float(v) * 0.44704, 3) if v is not None else None for v in wm]
                else:
                    try:
                        out["wind_m_s"] = float(out.get("wind_mph") * 0.44704)
                    except Exception:
                        pass
        # waves
        if "wave_height_m" not in out:
            if "wave_height_ft" in out:
                wf = out.get("wave_height_ft")
                if isinstance(wf, (list, tuple)):
                    out["wave_height_m"] = [round(float(v) * 0.3048, 3) if v is not None else None for v in wf]
                else:
                    try:
                        out["wave_height_m"] = float(out.get("wave_height_ft") * 0.3048)
                    except Exception:
                        pass
        # tide
        if "tide_height_m" not in out:
            if "tide_height_ft" in out:
                tf = out.get("tide_height_ft")
                if isinstance(tf, (list, tuple)):
                    out["tide_height_m"] = [round(float(v) * 0.3048, 3) if v is not None else None for v in tf]
                else:
                    try:
                        out["tide_height_m"] = float(out.get("tide_height_ft") * 0.3048)
                    except Exception:
                        pass
        # pressure
        if "pressure_hpa" not in out:
            if "pressure_inhg" in out:
                p = out.get("pressure_inhg")
                if isinstance(p, (list, tuple)):
                    out["pressure_hpa"] = [round(float(v) * 33.8638866667, 2) if v is not None else None for v in p]
                else:
                    try:
                        out["pressure_hpa"] = float(out.get("pressure_inhg") * 33.8638866667)
                    except Exception:
                        pass
        return out

    def validate(self, data: Dict[str, Any], species_profile: Any = None, units: str = "metric") -> Dict[str, Any]:
        """Validate and canonicalize remote payload, then build per-timestamp forecasts.

        - units: 'metric' or 'imperial' (will be converted into canonical metric fields)
        - species_profile: passed through to compute_score (string or dict)
        """
        # Preserve original payload for debugging
        raw_payload = dict(data)

        # Ensure timestamps present
        if "timestamps" not in data:
            raise ValueError("Missing timestamps in fetched data")
        timestamps = data.get("timestamps") or []
        if not isinstance(timestamps, (list, tuple)):
            raise ValueError("timestamps must be a list of ISO strings")
        n = len(timestamps)

        # Convert units if requested
        if units == "imperial":
            data = self._convert_imperial_to_metric(data)

        # Validate array lengths for known keys when present
        keys_to_check = ["temperature_c", "wind_m_s", "pressure_hpa", "wave_height_m", "tide_height_m"]
        for k in keys_to_check:
            val = data.get(k)
            if val is not None and isinstance(val, (list, tuple)) and len(val) != n:
                raise ValueError(f"Length of {k} ({len(val)}) does not match timestamps ({n})")

        # Build per-timestamp forecasts using compute_score (index-aware)
        forecasts = []
        for i in range(n):
            try:
                score = compute_score(data, species_profile=species_profile, use_index=i)
                forecasts.append({
                    "timestamp": timestamps[i],
                    "score_10": score.get("score_10"),
                    "score_100": score.get("score_100"),
                    "components": score.get("components", {}),
                })
            except MissingDataError:
                forecasts.append({"timestamp": timestamps[i], "score_10": None, "score_100": None, "components": {}, "error": "missing_inputs"})
            except Exception:
                forecasts.append({"timestamp": timestamps[i], "score_10": None, "score_100": None, "components": {}, "error": "compute_error"})

        result = dict(data)
        result["forecasts"] = forecasts
        result["raw_payload"] = raw_payload
        result["units"] = units
        result["species_profile_used"] = species_profile
        return result