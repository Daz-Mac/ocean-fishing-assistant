"""
Thin wrapper to validate and canonicalize data immediately after fetch.
This keeps the fetcher focused on retrieval and ensures downstream modules
receive a stable SI-structured payload.
"""
from typing import Dict, Any
from .ocean_scoring import compute_score, MissingDataError

class DataFormatter:
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure timestamps present
        if "timestamps" not in data:
            raise ValueError("Missing timestamps in fetched data")
        timestamps = data.get("timestamps") or []
        if not isinstance(timestamps, (list, tuple)):
            raise ValueError("timestamps must be a list of ISO strings")
        n = len(timestamps)

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
                score = compute_score(data, use_index=i)
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

        data["forecasts"] = forecasts
        return data