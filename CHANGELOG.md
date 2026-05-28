# Changelog

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
