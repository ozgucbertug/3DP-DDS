Grasshopper Python 3
====================

DDS can be used inside Rhino 8 Grasshopper Python 3 as a simulation backend.
The recommended workflow keeps Rhino and Grasshopper responsible for previews
while DDS creates domains, deposits, simulation fields, and extracted meshes.

Install the Grasshopper-focused extra when preparing a workshop environment:

.. code-block:: bash

   python -m pip wheel ".[gh]" -w wheelhouse

The ``gh`` extra includes ``scikit-image`` for marching-cubes mesh extraction.
It intentionally avoids the visualization, CLI, and YAML extras.

The adapter package is ``dds.gh_helpers``. It can be imported outside Rhino, but
functions that create Rhino geometry import RhinoCommon lazily and raise a clear
runtime error outside Rhino.

Recommended component flow:

* ``DDS Setup`` adds a local repository ``src`` path for editable workshop
  development. When DDS is installed into Rhino Python, this component can be
  skipped.
* ``DDS Domain Box`` converts a Rhino box or bounding box to ``dds.Domain``.
* ``DDS BeadProfile`` creates a reusable ``dds.BeadProfile``.
* ``DDS Target`` converts a point, plane/frame, or existing target to
  ``dds.DepositionTarget``. Points use world ``+Z`` when no normal is supplied.
* Point, line, and polyline deposit components consume target-like inputs plus
  ``dds.BeadProfile``.
* ``DDS Simulate`` combines the domain and deposits into ``SimulationResult``.
* ``DDS Mesh`` converts the result surface to ``Rhino.Geometry.Mesh``.

DDS objects should be passed directly through Grasshopper wires. Use sticky
state only for setup and optional simulation caching; do not append deposits to
sticky state during recompute.
