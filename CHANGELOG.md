# Changelog

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
