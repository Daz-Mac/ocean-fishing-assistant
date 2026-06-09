# Changelog

## [0.2.11] - 2026-06-09

### Added
- Spring tide bonus: all forecast periods on full moon and new moon days now
  receive a flat +10 point bonus (capped at 100) reflecting the stronger tidal
  movement from spring tides. The bonus is applied post-scoring, visible in
  sensor attributes (`spring_tide_bonus`), and displayed in the Lovelace card
  detail popup. Effects are universal — not species-specific.
- Card now shows a clear error state when the Open-Meteo or World Tides API is
  unreachable, instead of displaying NaN. If cached forecast data is available,
  the card shows it with an error banner; otherwise a minimal error card with
  troubleshooting hints is displayed.
- Forecast rows now display safety breach factor names (e.g. "⚠ Wind") inline on
  the main card when the score is locked at 30/100 due to unsafe conditions.
  Clean rows show no extra text. The detail popup also shows a safety line
  explaining which factors were breached.

## [0.2.10] - 2026-06-07

### Fixed
- Period tide component now uses peak score instead of average — a 6-hour
  period containing a high tide shows the best tide window within it rather
  than diluting across all hours
- Added "new moon" to the default general profile's moon preferences alongside
  "full moon"

## [0.2.9] - 2026-06-06

### Fixed
- Species profiles with `preferred_tide_phase: ["high"]` (including the default
  general profile) now correctly score proximity to high tide instead of always
  returning 30/100. TideProxy now exposes each timestamp's distance to the
  nearest high/low extreme, and the scoring engine uses this for proximity-based
  scoring with a ±1.5-hour window.

### Added
- Tide proximity scoring: species with `"high"` or `"low"` tide preferences now get
  scored based on how close they are to the actual high/low tide event (within a
  ±1.5-hour window), rather than a flat 30/100
- 11 new unit tests covering tide proximity scoring for high/low windows,
  combined phase+proximity preferences, and backward-compatible fallback
- Tide score bar and numeric value in the forecast row detail popup in the
  Lovelace custom card

### Fixed
- Period tide component now uses peak score instead of average — a 6-hour
  period containing a high tide shows the best tide window within it rather
  than diluting across all hours
- Added "new moon" to the default general profile's moon preferences alongside
  "full moon"

## [0.2.8] - 2026-06-03

### Fixed
- Blueprint directory listing during integration setup no longer triggers "Detected blocking call to listdir" in Home Assistant logs — synchronous `os.listdir()` calls are now properly offloaded to the async executor

## [0.2.7] - 2026-05-31

### Fixed
- Tide arrows (↑/↓) removed from forecast rows — unclear to end users. Tide phase still visible when tapping a row for details
- Forecast detail popup no longer closes by itself after ~1 second on sensor refresh — expanded state is now preserved across Home Assistant re-renders
- Restored missing `tomorrow` variable in daily briefing blueprint that caused "tomorrow is undefined" error

## [0.2.6] - 2026-05-29

### Added
- Scoring engine test coverage — 116 tests covering all 8 factors, safety capping, breach detection, and weight normalization
- Data formatter test coverage — 22 tests covering payload validation, unit conversion, and period building
- CI pipeline now runs all tests on every push and PR
- Browser console debug logging is now tied to the "Enable raw data" config option — off by default, on when debugging

### Fixed
- Swell safety capping was silently broken — `_SAFETY_COMPONENT_MAP` prefix "swell" didn't match the reason code "swell_period<8.0"
- Unnecessary browser console messages removed from the Lovelace custom card
- All three notification blueprints now define `sensor` in a top-level `variables` block so condition templates can resolve it — previously conditions failed with "sensor is undefined" on trigger

## [0.2.5] - 2026-05-29

### Fixed
- Notification period labels no longer show malformed text (e.g. "h18h-24" → "18-24h") in daily briefing
- Duplicated "Today's Fishing Forecast" title removed from daily briefing notification body
- Period lines now render on separate lines instead of concatenating in daily briefing and prime conditions notifications
- `{%- endif -%}` changed to `{%- endif %}` inside for-loop templates to preserve newlines between iterations

### Changed
- Updated GitHub Actions workflows to Node.js 24 compatible versions (actions/checkout@v6, softprops/action-gh-release@v3)

## [0.2.4] - 2026-05-28

### Fixed
- All three notification blueprints now also create a persistent HA notification so the full message content is viewable after tapping the push notification (daily_briefing, score_alert, prime_conditions)

## [0.2.2] - 2026-05-28

### Fixed
- Blueprint message templates now use `variables:` block to fix "sensor is undefined" Jinja2 error on notify actions
- Notification target switched from text input to device selector dropdown showing registered mobile devices

## [0.2.1] - 2026-05-28

### Fixed
- Blueprints now deploy correctly on new HACS installations — moved inside `custom_components/` with auto-copy to HA blueprint directory on integration startup

## [0.2.0] - 2026-05-28

### Added
- Notification blueprints — 3 Home Assistant automation blueprints for score alerts, daily fishing briefing, and prime conditions detection
- YAML validation of blueprints added to CI pipeline
- User documentation: getting-started guide, score explanation, safety configuration, and sensor attribute reference
- World Tides API cost breakdown and pricing recommendation in README headings
- TODO list tracking for integration development

## [0.1.1] - 2026-05-27

### Added
- HACS brand assets (`brand/icon.png` + `brand/icon@2x.png`) for integration listing
- HACS validation workflow (`.github/workflows/validate.yml`) — auto-validates on push and PR
- Home Assistant validation workflow (`.github/workflows/hassfest.yml`) — catches manifest issues
- Release automation workflow (`.github/workflows/release.yml`) — auto-creates release on tag push
- Apache 2.0 `LICENSE` file matching the project's stated license
- `my.home-assistant.io` one-click install badge in README
- HACS validation and hassfest status badges in README
- Release process documentation in README

### Changed
- Cleaned up `hacs.json` (removed `render_readme` field)
- Bumped version to 0.1.1
