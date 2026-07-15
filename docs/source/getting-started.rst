Getting Started
===============

Installation
------------

3DP-DDS requires Python 3.9 or newer. From a local clone, install the core
package in editable mode:

.. code-block:: bash

   python -m pip install -e .

Optional extras are installed only when needed:

.. list-table::
   :header-rows: 1

   * - Extra
     - Capability
   * - ``cli``
     - Tyro-backed helpers for typed example CLIs
   * - ``formats``
     - YAML target loading
   * - ``mesh``
     - Mesh I/O, mesh conversion, and mesh-backed signed-distance queries
   * - ``viz``
     - Interactive PyVistaQt viewer and simulation workbench
   * - ``all``
     - All optional capabilities

First Simulation
----------------

.. doctest::

   >>> from dds import BeadProfile, Domain, LineDeposit, simulate
   >>> profile = BeadProfile(width=1.2, height=0.6)
   >>> deposits = [
   ...     LineDeposit(
   ...         start=(0.0, 0.0, 0.6),
   ...         end=(2.0, 0.0, 0.6),
   ...         profile=profile,
   ...     )
   ... ]
   >>> domain = Domain.from_deposits(deposits, voxel_size=0.5, padding="auto")
   >>> result = simulate(domain, deposits, threshold=0.5)
   >>> result.implicit_field.shape == domain.grid_shape
   True
   >>> result.analysis.occupancy().dtype.name
   'bool'

Next Steps
----------

Read the tutorials for modeling workflow, then use the generated API reference
for exact signatures and parameters.
