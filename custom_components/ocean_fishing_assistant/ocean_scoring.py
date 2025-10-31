"""
Scoring module. Expects data in canonical SI units (see fetcher/formatter).
Returns score (0-100) and component breakdown dictionary.
"""
from typing import Dict, Any

def compute_score(data: Dict[str, Any], species_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    # Basic stub scoring: returns a midpoint score if data present.
    if not data or "timestamps" not in data:
        raise ValueError("Missing data to compute score")

    # Placeholder: more advanced scoring adapted from examples will be implemented here.
    score = 50
    components = {
        "temperature_factor": 1.0,
        "wind_factor": 1.0,
        "wave_factor": 1.0,
    }

    return {"score": score, "components": components}