# Modeling Assumptions

3DP-DDS models deposited geometry, not coupled process physics.

## Geometry model

- Deposit targets are top-referenced nozzle positions.
- `Pose3D` is an active local-to-parent rigid transform backed by
  `scipy.spatial.transform.Rotation`.
- A pose becomes a deposition target by transforming tool-local `+Z`, unless
  the caller explicitly supplies another local axis.
- Tool roll is discarded after conversion because the current bead profile is
  rotationally symmetric about the deposition normal.
- `BeadProfile.width` and `BeadProfile.height` define explicit world-space bead
  geometry and are required for every deposit.
- A point is one bead target, a line is a swept bead, and a polyline is one
  ordered multi-segment fabrication event.
- Endpoint normals are normalized and interpolated along line segments.
- Antiparallel endpoint normals are rejected because interpolation is ambiguous
  without an intermediate orientation.
- The implicit field is the union-like fabricated geometry used for
  occupancy, surface extraction, SDF construction, and support analysis.
- The stored implicit field is not itself a signed-distance field. It is
  bounded to `[0, 1]`, is nonnegative, and uses `0.5` as the nominal surface.
  `SimulationAnalysis.surface_sdf()` derives a signed, world-unit distance
  representation when metric distance or surface normals are needed.
- Changing voxel size changes discretization, not the specified bead support
  bounds or other world-space geometry.

## Coverage diagnostic

Coverage adds anti-aliased kernel values. It can locate path overlap, but it is
not mass, volume fraction, material density, or deposited flow. Coverage may
depend on discretization and path segmentation and should not be compared
across resolutions without a separate calibration model.

## Units

`Domain.length_unit` is either `"mm"` or `"m"` and documents the unit used by
world coordinates, bead dimensions, and voxel size. The library does not
convert units.

## Current exclusions

The simulator does not model:

- material flow conservation or extrusion transients;
- gravity, sagging, curing, cooling, or thermal history;
- collision, reachability, robot dynamics, or controller timing;
- bead deformation caused by substrate contact or previous deposits;
- uncertainty propagation or experimental parameter calibration.

## Numerical interpretation

Occupancy is obtained by thresholding the implicit field. Research results
should report the domain, `length_unit`, voxel size, threshold, bead profile,
and path definition. Convergence studies across voxel sizes are recommended
for quantitative use.
