Fields And Thresholds
=====================

The implicit field is the canonical fabricated geometry envelope. It is
nonnegative, clipped to ``[0, 1]``, and composed with maximum operations.

Occupancy is obtained by thresholding the implicit field. The default threshold
is commonly ``0.5``.

Coverage is additive and useful for locating overlap. It is not physical mass,
density, volume fraction, or flow.
