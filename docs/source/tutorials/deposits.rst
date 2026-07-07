Deposits
========

Deposits combine top-referenced targets with bead dimensions. Coordinate
triplets are interpreted as world ``+Z`` targets.

.. code-block:: python

   from dds import BeadProfile, DepositionTarget, LineDeposit, PointDeposit

   profile = BeadProfile(width=1.2, height=0.6)

   point = PointDeposit(target=(2.0, 2.0, 0.6), profile=profile)
   line = LineDeposit(
       start=DepositionTarget((2.0, 2.0, 0.6), normal=(0.0, 0.0, 1.0)),
       end=DepositionTarget((10.0, 2.0, 0.6), normal=(0.0, 0.0, 1.0)),
       profile=profile,
   )

Use :class:`dds.PolylineDeposit` for one ordered multi-segment fabrication
event.
