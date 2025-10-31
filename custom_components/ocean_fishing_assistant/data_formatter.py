"""
Thin wrapper to validate and canonicalize data immediately after fetch.
This keeps the fetcher focused on retrieval and ensures downstream modules
receive a stable SI-structured payload.
"""
from typing import Dict, Any

class DataFormatter:
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Basic validation: ensure numeric arrays / keys exist, otherwise raise ValueError
        if "timestamps" not in data:
            raise ValueError("Missing timestamps in fetched data")
        # Additional validation/canonicalization can be added here
        return data