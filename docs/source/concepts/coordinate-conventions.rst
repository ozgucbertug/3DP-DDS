Coordinate Conventions
======================

3DP-DDS uses world-space coordinates in ``(x, y, z)`` order. Dense arrays use
the same axis order and NumPy ``indexing="ij"``.

Targets are top-referenced. A deposition target is usually a nozzle target at
the top of the bead, not the bead center. Bead height extends opposite the
target normal.
