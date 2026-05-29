# TODOs

## T1 — Scoring engine test coverage (COMPLETE)
- [x] Design doc approved via /office-hours
- [x] Eng review complete via /plan-eng-review
- [x] Create `tests/_scoring_bootstrap.py` — shared import bootstrap
- [x] Create `tests/test_scoring.py` — scoring engine tests (all factors, safety capping, breaches)
- [x] Create `tests/test_formatter.py` — data formatter tests (validation, unit conversion, period building)
- [x] Fix `tests/conftest.py` — remove zoneinfo from mock list (stdlib, always available)
- [x] Fix production bug: `_SAFETY_COMPONENT_MAP` prefix "swell" → "swell_period" (silently failed to match reason code "swell_period<8.0")
- [x] Add `pytest` step to `.github/workflows/validate.yml`
- [x] Run all tests and verify — **160 tests pass** (22 safety + 116 scoring + 22 formatter)

## T2 — CI test runner (COMPLETE, delivered with T1)
- [x] Add `pytest` step to `.github/workflows/validate.yml`
- [x] Run existing `tests/test_safety.py` and new scoring/formatter tests on every push/PR
- [x] Keep CI fast — scoring tests are pure logic, no network needed