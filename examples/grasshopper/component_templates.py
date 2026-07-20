# ruff: noqa
"""Copyable Grasshopper Python 3 component templates for DDS.

Each section is meant to be copied into a separate GH Python component with the
listed input and output names.
"""


# DDS Setup
# Inputs: package_path, run
# Outputs: DDSPath
# r: numpy==2.0.2, scipy==1.13.1, scikit-image==0.24.0
#
# - package_path points at the repository `src` folder, not the repository root.
# - Use a Button for run. The path is stored in scriptcontext.sticky so it
#   remains available after the button releases.
# - This only points Rhino Python at DDS source code. It does not install
#   runtime dependencies such as numpy, scipy, or scikit-image.
# - DDSPath should resolve to `.../3DP-DDS/src/dds/__init__.py`; any other path
#   means Rhino is importing a different DDS copy.

import sys
import scriptcontext as sc

KEY = "dds.gh_helpers.source_path"

if run and package_path:
    sc.sticky[KEY] = package_path

DDS_SRC = sc.sticky.get(KEY, package_path)

if DDS_SRC and DDS_SRC not in sys.path:
    sys.path.insert(0, DDS_SRC)

OK = False
DDSPath = None

if DDS_SRC:
    try:
        import dds
        import dds.gh_helpers

        DDSPath = dds.__file__
        if bool(DDSPath and DDSPath.startswith(DDS_SRC)):
            print("DDS loaded from:", DDSPath)
        else:
            print("DDS imported from unexpected path:", DDSPath)
    except Exception as exc:
        DDSPath = None
        print("DDS setup failed:", repr(exc))
else:
    print("Press run with package_path set to the DDS src folder.")


# DDS Domain Box
# Inputs: Box, voxel_size
# Outputs: Domain, PreviewBox
#
# - Box is a Rhino box or any object with a bounding box.
# - DDS domains are axis-aligned dense voxel grids; rotated Rhino boxes are
#   converted through their world-axis bounding box.
# - voxel_size is in the active Rhino model units. The GH helper records this
#   as millimeters internally, but it does not convert units.
# - Bounds are expanded upward to whole voxels, so Domain.max_corner may be
#   slightly larger than the input box.
# - Runtime cost scales with grid_shape[0] * grid_shape[1] * grid_shape[2].
from dds.gh_helpers.components import make_domain_from_box

Domain = None
PreviewBox = None

try:
    Domain, PreviewBox = make_domain_from_box(Box, voxel_size)
    print("Domain:", Domain.grid_shape, "voxels")
except Exception as exc:
    print("Domain creation failed:", repr(exc))


# DDS Profile
# Inputs: width, height
# Outputs: Profile
#
# - width and height use the same units as the Domain and Target coordinates.
# - Profile describes the nominal bead cross-section used by all deposits wired
#   to it.
# - Targets are top/nozzle-referenced. The GH helpers do not expose
#   center-reference conversion.
from dds.gh_helpers.components import make_bead_profile

Profile = None

try:
    Profile = make_bead_profile(width, height)
    print("Profile:", Profile.width, "x", Profile.height)
except Exception as exc:
    print("Profile creation failed:", repr(exc))


# DDS Target
# Inputs: Position, Normal
# Outputs: Target
#
# - Position can be a Rhino point, Rhino plane/frame, or an existing DDS target.
# - If Position is a plane/frame, its origin and normal define the target and
#   the Normal input is ignored.
# - If Position is a point and Normal is empty, DDS uses world +Z.
# - Target is the only orientation carrier for point/line/polyline deposits.
from dds.gh_helpers.components import make_target

Target = None

try:
    Target = make_target(Position, normal=Normal)
    print("Target:", Target.position.to_tuple(), Target.normal.to_tuple())
except Exception as exc:
    print("Target creation failed:", repr(exc))


# DDS Target From Plane
# Inputs: Plane
# Outputs: Target
#
# - This is a convenience version of DDS Target for explicit plane/frame input.
# - The plane origin becomes target position.
# - The plane normal/ZAxis becomes deposition normal.
# - Roll around the normal is not used by DDS' current bead model.
from dds.gh_helpers.components import make_target_from_plane

Target = None

try:
    Target = make_target_from_plane(Plane)
    print("Target:", Target.position.to_tuple(), Target.normal.to_tuple())
except Exception as exc:
    print("Target from plane failed:", repr(exc))


# DDS Point Deposit
# Inputs: Target, Profile
# Outputs: Deposit
#
# - Target can be a DDS target, Rhino plane/frame, or Rhino point.
# - If Target is a point, world +Z is used as its normal.
# - The deposit represents one bead centered below the top/nozzle-referenced
#   target according to Profile.height.
from dds.gh_helpers.components import make_point_deposit

Deposit = None

try:
    Deposit = make_point_deposit(Target, Profile)
    print("PointDeposit created")
except Exception as exc:
    print("Point deposit failed:", repr(exc))


# DDS Line Deposit
# Inputs: StartTarget, EndTarget, Profile, sweep_resolution
# Outputs: Deposit
#
# - StartTarget and EndTarget can each be a DDS target, Rhino plane/frame, or
#   Rhino point.
# - If either endpoint is a point, world +Z is used for that endpoint normal.
# - Endpoint normals are interpolated along the line by DDS.
# - sweep_resolution is optional. Leave it empty for DDS to choose a
#   conservative sampling resolution from the domain and profile.
from dds.gh_helpers.components import make_line_deposit

Deposit = None

try:
    Deposit = make_line_deposit(
        StartTarget,
        EndTarget,
        Profile,
        sweep_resolution=sweep_resolution or None,
    )
    print("LineDeposit length:", Deposit.line.length)
except Exception as exc:
    print("Line deposit failed:", repr(exc))


# DDS Polyline Deposit
# Inputs: Targets, Profile, sweep_resolution
# Outputs: Deposit
#
# - Targets should use List Access.
# - Each item can be a DDS target, Rhino plane/frame, or Rhino point.
# - Points default to world +Z; planes carry their own normals.
# - DDS creates one continuous multi-segment bead through the ordered targets.
# - Use separate deposits instead if you need separated toolpath events.
from dds.gh_helpers.components import make_polyline_deposit

Deposit = None

try:
    Deposit = make_polyline_deposit(
        Targets,
        Profile,
        sweep_resolution=sweep_resolution or None,
    )
    print("PolylineDeposit targets:", len(Deposit.targets))
except Exception as exc:
    print("Polyline deposit failed:", repr(exc))


# DDS Simulate
# Inputs: Domain, Deposits, run, reset, include_coverage
# Outputs: Result
#
# - Deposits should use List Access and may receive one or many deposit objects.
# - run gates the expensive dense-field computation. Keep it False while
#   editing sliders, then set True when you want a result.
# - reset clears the optional sticky result cache used by dds.gh_helpers.
# - include_coverage=True computes an additive overlap diagnostic. Keep it
#   False for normal mesh preview.
# - threshold is fixed to 0.5 in this template for a simpler GH UI.
# - The output Result is immutable; recomputing does not append duplicate
#   deposits.
from dds.gh_helpers.components import run_simulation

Result = None

try:
    Result = run_simulation(
        Domain,
        Deposits,
        run=run,
        reset=reset,
        include_coverage=include_coverage,
        threshold=0.5,
    )
    if Result is None:
        print("Simulation ready. Set run=True to compute.")
    else:
        print("Simulation deposits:", len(Result.deposits))
except Exception as exc:
    print("Simulation failed:", repr(exc))


# DDS Mesh
# Inputs: Result, threshold, step_size
# Outputs: Mesh
#
# - Mesh extraction runs marching cubes on Result.implicit_field.
# - This requires scikit-image in Rhino's Python environment.
# - threshold defaults to the Result threshold when empty.
# - step_size=1 gives best quality. Larger values, such as 2 or 3, are faster
#   preview meshes.
# - Grasshopper/Rhino handles display; DDS does not import PyVista or PySide.
from dds.gh_helpers.components import make_mesh

Mesh = None

try:
    if Result is None:
        print("No Result input.")
    else:
        Mesh = make_mesh(
            Result,
            threshold=threshold or None,
            step_size=step_size or 1,
        )
        print("Mesh created")
except Exception as exc:
    print("Mesh extraction failed:", repr(exc))
