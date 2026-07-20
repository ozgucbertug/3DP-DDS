# DDS in Grasshopper Python 3

These scripts are intended for Rhino 8 Grasshopper Python 3 components.
Use DDS as the simulation backend and Rhino/Grasshopper geometry as the preview
layer.

## Install Options

For editable workshop development, use `DDS Setup` to add the local repository
`src` path inside Grasshopper. This avoids building or installing a DDS wheel
while you are iterating on the source.

If DDS is installed into Rhino's Python environment, `DDS Setup` is optional.
Install the `mesh` extra only when using the `DDS Mesh` component:

```bash
python -m pip install "3dp-dds[mesh]"
```

Grasshopper and Rhino handle visualization. DDS' `viz` extra is for the separate
PyVista/PySide workbench and is not needed for this workflow.

## Component Dataflow

Use Python objects through Grasshopper wires:

1. `DDS Domain Box`: Rhino Box -> `dds.Domain`
2. `DDS BeadProfile`: width/height -> `dds.BeadProfile`
3. `DDS Target`: Rhino point or plane/frame -> `dds.DepositionTarget`
4. `DDS Point Deposit`, `DDS Line Deposit`, or `DDS Polyline Deposit`: target-like inputs + profile -> DDS deposit
5. `DDS Simulate`: domain + deposits -> `dds.SimulationResult`
6. `DDS Mesh`: result -> Rhino mesh preview

The components rebuild from current inputs. Sticky storage is only used for
optional simulation caching and setup diagnostics, avoiding duplicate deposits
on canvas recompute.
