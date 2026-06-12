# Changelog

All notable changes will be documented in this file.

## Unreleased

### Added

- Point, line, and first-class polyline deposition events with nozzle poses.
- Explicit process state, unit systems, dense fields, and chunked fields.
- Result checkpointing, headless analysis, YAML target loading, and examples.
- Repository validation, contribution guidance, and modeling documentation.
- A retained PyVistaQt viewer for meshes, points, paths, poses, deposition
  targets, and grouped deposit overlays.
- Immutable point clouds with trimesh conversion, file I/O, optional
  per-point colors, and retained-viewer support.

### Changed

- Max-envelope density is the canonical geometry field.
- Additive coverage is explicitly identified as a nonphysical diagnostic.
- The simulation workbench now consumes the retained viewer for optional
  toolpath, target-normal, and world-frame overlays.
- Mesh and point-cloud conversion and file I/O now consistently use trimesh.
- Mesh metrics, normals, proximity queries, metadata, and visual colors now
  delegate to trimesh rather than duplicating geometry calculations.

### Fixed

- Domain alignment, deposit bounds, serialization, metadata immutability, and
  orientation interpolation invariants.
