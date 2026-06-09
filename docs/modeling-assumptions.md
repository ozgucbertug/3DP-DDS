# Modeling Assumptions

3DP-DDS currently models deposited geometry, not coupled process physics.

## Geometry model

- Deposit targets are top-referenced nozzle positions.
- `BeadProfile.width` and `BeadProfile.height` define a nominal compact bead.
- A point is one bead target, a line is a swept bead, and a polyline is one
  ordered multi-segment fabrication event.
- Endpoint bead axes are normalized and interpolated along line segments.
- Antiparallel endpoint axes are rejected because their interpolation is not
  unique without an intermediate orientation.
- The `"max"` field is the union-like geometric envelope used for occupancy,
  surface extraction, and analysis.

## Coverage diagnostic

The `"coverage"` field adds anti-aliased kernel values. It is useful for
locating path overlap, but it is not mass, volume fraction, material density,
or deposited flow. Its values depend on voxel resolution and path
segmentation, so it should not be compared across discretizations without a
separate calibration model.

## Current exclusions

The simulator does not yet model:

- material flow conservation or extrusion transients;
- gravity, sagging, curing, cooling, or thermal history;
- collision, reachability, robot dynamics, or controller timing;
- bead deformation caused by substrate contact or previous layers;
- uncertainty propagation or experimental parameter calibration.

`ProcessState` records process inputs for provenance and future models. Those
values do not currently modify the geometric kernel.

## Numerical interpretation

Occupancy is obtained by thresholding the max-envelope field. Results should
include the domain, voxel size, threshold, bead profile, and path definition
when reported. Convergence studies across voxel sizes are recommended for
quantitative research use.
