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
- [ ] Create `blueprints/automation/ocean_fishing_assistant/` with a Home Assistant automation blueprint
- [ ] Trigger: fishing score above/below threshold (e.g., score > 70)
- [ ] Actions: notify mobile device, with score breakdown in message
- [ ] Include species name, score, top factors, and safety warnings in notification
