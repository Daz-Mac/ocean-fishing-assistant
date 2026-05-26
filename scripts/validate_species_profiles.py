#!/usr/bin/env python3
"""Validate species_profiles.json against its schema and semantic rules.

Usage:
    python3 scripts/validate_species_profiles.py [--profiles PATH] [--schema PATH]

Exits with code 0 if valid, 1 on validation errors.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_schema(profiles_path, schema_path):
    """Validate JSON structure against schema."""
    try:
        import jsonschema
    except ImportError:
        print("SKIP: jsonschema not installed — install with: pip install jsonschema")
        return True

    try:
        profiles = load_json(profiles_path)
        schema = load_json(schema_path)
        jsonschema.validate(profiles, schema)
        print(f"  Schema: OK")
        return True
    except jsonschema.exceptions.ValidationError as e:
        print(f"  Schema ERROR: {e.message}")
        print(f"    Path: {'.'.join(str(p) for p in e.absolute_path)}")
        return False
    except Exception as e:
        print(f"  Schema ERROR: {e}")
        return False


def validate_semantic(profiles_path):
    """Validate semantic rules beyond schema constraints."""
    errors = []
    profiles = load_json(profiles_path)
    species = profiles.get("species", {})
    regions = profiles.get("regions", {})
    general_profiles = profiles.get("general_profiles", {})

    # Every species must have valid region references
    for sid, s in species.items():
        for r in s.get("regions", []):
            if r not in regions:
                errors.append(f"  ERROR: species '{sid}' references unknown region '{r}'")

    # General profiles may use "global" as a sentinel for all-regions
    known_regions = set(regions.keys()) | {"global"}
    for gid, g in general_profiles.items():
        for r in g.get("regions", []):
            if r not in known_regions:
                errors.append(f"  ERROR: general_profile '{gid}' references unknown region '{r}'")

    # Temperature ranges must have min <= max when both provided
    for sid, s in species.items():
        tc = s.get("preferred_temp_c", [])
        if len(tc) == 2 and tc[0] is not None and tc[1] is not None:
            if tc[0] > tc[1]:
                errors.append(f"  ERROR: species '{sid}' preferred_temp_c min > max ({tc[0]} > {tc[1]})")

    # Wind ranges must have min <= max
    for sid, s in species.items():
        w = s.get("preferred_wind_m_s", [])
        if len(w) == 2 and w[0] is not None and w[1] is not None:
            if w[0] > w[1]:
                errors.append(f"  ERROR: species '{sid}' preferred_wind_m_s min > max ({w[0]} > {w[1]})")

    # swell period ranges
    for sid, s in species.items():
        sp = s.get("preferred_swell_period_s", [])
        if len(sp) == 2 and sp[0] is not None and sp[1] is not None:
            if sp[0] > sp[1]:
                errors.append(f"  ERROR: species '{sid}' preferred_swell_period_s min > max ({sp[0]} > {sp[1]})")

    # preferred_times: verify hour ranges
    valid_time_tokens = {"dawn", "dusk"}
    for sid, s in species.items():
        for i, t in enumerate(s.get("preferred_times", [])):
            if isinstance(t, dict):
                sh = t.get("start_hour")
                eh = t.get("end_hour")
                if sh is not None and eh is not None:
                    if not (0 <= int(sh) <= 23 and 0 <= int(eh) <= 23):
                        errors.append(f"  ERROR: species '{sid}' preferred_times[{i}] hour out of range (0-23)")
            elif isinstance(t, str):
                if t.lower() not in valid_time_tokens:
                    errors.append(f"  ERROR: species '{sid}' preferred_times[{i}] invalid token '{t}' (use da|dusk)")

    # Wave height must be positive
    for sid, s in species.items():
        mw = s.get("max_wave_height_m")
        if mw is not None and isinstance(mw, (int, float)) and mw <= 0:
            errors.append(f"  ERROR: species '{sid}' max_wave_height_m must be > 0 (got {mw})")

    if errors:
        print("  Semantic checks:")
        for e in errors:
            print(f"    {e}")
        return False

    print("  Semantic: OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate species_profiles.json")
    script_dir = Path(__file__).resolve().parent
    default_profiles = script_dir.parent / "custom_components" / "ocean_fishing_assistant" / "species_profiles.json"
    default_schema = script_dir.parent / "custom_components" / "ocean_fishing_assistant" / "species_schema.json"

    parser.add_argument("--profiles", default=str(default_profiles), help="Path to species_profiles.json")
    parser.add_argument("--schema", default=str(default_schema), help="Path to species_schema.json")
    args = parser.parse_args()

    profiles_path = args.profiles
    schema_path = args.schema

    if not os.path.exists(profiles_path):
        print(f"ERROR: profiles file not found: {profiles_path}")
        sys.exit(1)

    print(f"Validating: {profiles_path}")
    schema_ok = validate_schema(profiles_path, schema_path)
    semantic_ok = validate_semantic(profiles_path)

    species = load_json(profiles_path).get("species", {})
    regions = load_json(profiles_path).get("regions", {})
    general = load_json(profiles_path).get("general_profiles", {})
    print(f"  Summary: {len(species)} species, {len(regions)} regions, {len(general)} general profiles")

    if schema_ok and semantic_ok:
        print("RESULT: VALID")
        sys.exit(0)
    else:
        print("RESULT: INVALID")
        sys.exit(1)


if __name__ == "__main__":
    main()
