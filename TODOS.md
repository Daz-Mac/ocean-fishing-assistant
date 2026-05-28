# TODOs

## T1 — Scoring engine test coverage
- [ ] Add tests for `ocean_scoring.py` — weighted scoring, factor calculations, safety capping
- [ ] Add tests for `data_formatter.py` — payload validation, score aggregation, period assembly
- [ ] `conftest.py` already mocks HA dependencies; test structure is established

## T2 — CI test runner
- [ ] Add `pytest` step to `.github/workflows/validate.yml`
- [ ] Run existing `tests/test_safety.py` and new scoring/formatter tests on every push/PR
- [ ] Keep CI fast — scoring tests are pure logic, no network needed

## T6 — Notification blueprint
- [x] Create `blueprints/automation/ocean_fishing_assistant/` with 3 blueprints:
  - `score_alert.yaml` — rising-edge threshold alert with quiet hours
  - `daily_briefing.yaml` — morning period forecast briefing
  - `prime_conditions.yaml` — high score + no breaches alert
- [x] YAML validation in CI (validate.yml)
- [x] Include species name, score, factor breakdown, quiet hours, availability guards
