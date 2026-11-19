"""Compare the Fermi-surface and Fermi-sea Hall conductivity of a Weyl model.

Requires QuantumGeom (imported as quantumGeom), NumPy, and Matplotlib.
From the repository root, run:

    python examples/weyl_hall_conductivity.py

This writes a PNG and a CSV in the current directory. The bundled
weyl_semi_tb.dat supplies the Wannier90 tight-binding model. Increase the mesh
sizes and decrease the energy step for a finer scan, for example:

    python examples/weyl_hall_conductivity.py --step 0.05 --points 4000 --sea-points 80000 --bz-grid 512

The plot compares the native QuantumGeom Fermi-sea and Fermi-surface results
with an independent reference from the analytic two-band Hamiltonian. The
reference uses midpoint quadrature at finite Fermi energies and the exact
Chern-slice result at the Weyl nodes. All curves are in units of e^2/(2ha).
The Fermi-surface result includes the Berry phase across the periodic cuts. Its
Hall branch can still differ from the Fermi-sea result by whole Hall quanta of
e^2/(ha). The CSV includes the surface-minus-sea difference modulo that quantum
to diagnose numerical convergence while keeping all three curves unshifted.
See Haldane, Eq. (21), https://arxiv.org/abs/cond-mat/0408417.
"""

import argparse
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
from xml.etree import ElementTree

import matplotlib
import numpy as np
import quantumGeom as qg

matplotlib.use("Agg")  # Save a figure even on a machine without a display.
import matplotlib.pyplot as plt


MODEL = Path(__file__).with_name("weyl_semi_tb.dat").resolve()
LATTICE_SPACING_ANGSTROM = 1.0
BAND_EDGE_EV = 10.0
# Both native tensors have units of inverse Angstrom. Convert M_yz with
# sigma_yz = -(e^2/hbar) M_yz/(2*pi)^3 and divide by e^2/(2*h*a).
HALL_RATIO_SCALE = -LATTICE_SPACING_ANGSTROM / (2 * np.pi**2)
HALL_QUANTUM_RATIO = 2.0


def encloses_both_nodes(mesh):
    """Check the oriented solid angle of the exported triangles at each node."""
    points = np.fromstring(mesh.find("Points/DataArray").text, sep=" ").reshape(-1, 3)
    indices = np.fromstring(
        mesh.find("Polys/DataArray[@Name='connectivity']").text, sep=" ", dtype=int
    ).reshape(-1, 3)
    triangles = points[indices]
    # VTP coordinates are Cartesian, so the nodes at reduced kx=±1/4 are here.
    node_kx = np.pi / (2 * LATTICE_SPACING_ANGSTROM)
    for x in (-node_kx, node_kx):
        rays = triangles - [x, 0, 0]
        lengths = np.linalg.norm(rays, axis=2, keepdims=True)
        if np.any(lengths == 0):
            return False
        a, b, c = (rays / lengths).transpose(1, 0, 2)
        triple_product = np.sum(a * np.cross(b, c), axis=1)
        dot_sum = np.sum(a * b + b * c + c * a, axis=1)
        solid_angle = 2 * np.arctan2(triple_product, 1 + dot_sum).sum()
        if not np.isclose(abs(solid_angle), 4 * np.pi, atol=1e-4):
            return False
    return True


def fermi_surface_ratio(bulk, energy, mesh_file, points, refinements, threads):
    """Return QuantumGeom's yz Hall result in units of the Weyl-node reference."""
    # Band indices start at zero. The bands occupy [-10, 0] and [0, 10] eV.
    band = 0 if energy < 0 else 1
    near_nodes = abs(energy) <= 0.35
    if near_nodes or abs(energy) > 9.9:
        points, refinements = max(points, 10_000), max(refinements, 4)

    # Start a fresh periodic Delaunay mesh at each energy, then increase the
    # resolution if a small pocket was missed by the random initial mesh.
    for initial_points, rounds in ((points, refinements), (2 * points, refinements + 1)):
        bulk.initializeTheMeshParallel(initial_points, threads)
        bulk.refineTheMeshForOneBandParallel(rounds, band, energy, threads)

        surface = qg.FermiSurface(energy, band)
        surface.generateSurface(bulk)

        # The VTP gives the face count and lets us check that both tiny pockets
        # are present near zero. The four flags disable optional data and use ASCII.
        surface.writeSurfaceVtp(str(mesh_file), False, False, False, False)
        mesh = ElementTree.parse(mesh_file).find(".//Piece")
        if int(mesh.attrib["NumberOfPolys"]) == 0 or (near_nodes and not encloses_both_nodes(mesh)):
            continue

        # Berry phases use the eigenstates on the vertices of each triangle.
        surface.obtainVertexEigenstatesParallel(bulk, threads)
        surface.obtainBerryCurvature()
        haldane_tensor = surface.intrinsicMatrixOfHaldaneIntegral()

        return HALL_RATIO_SCALE * haldane_tensor[1, 2]

    warnings.warn(f"Could not resolve the full Fermi surface at E_F={energy:g} eV; leaving a gap.",
                  RuntimeWarning, stacklevel=2)
    return np.nan


def occupied_state_ratios(positive_energies, grid_size):
    """Return the model reference at sorted nonnegative Fermi energies in eV."""
    # In angular coordinates x,y,z = 2*pi*k_reduced, the input file gives
    # H = 2*d·sigma in eV, with d = (m - cos(y) - cos(z), -sin(y), -sin(z))
    # and m = 2 + cos(x). The lower band's yz Berry curvature is
    # d · (∂_y d × ∂_z d) / (2|d|^3). The bands have energies ±2|d| and
    # opposite curvatures. For 2|d| < |E_F| both bands are occupied on the
    # electron side and both are empty on the hole side, giving zero in
    # either case. Only the lower band contributes where 2|d| > |E_F|.
    # Reflection symmetry lets a midpoint grid on [0, pi]^3 represent the BZ.
    angles = (np.arange(grid_size) + 0.5) * np.pi / grid_size
    cosines = np.cos(angles)
    cos_y, cos_z = cosines[:, None], cosines[None, :]
    cos_sum = cos_y + cos_z
    cos_product = cos_y * cos_z
    sin_sq = np.sin(angles)**2
    transverse_sq = sin_sq[:, None] + sin_sq[None, :]
    by_energy = np.zeros(positive_energies.size + 1)

    for mass in 2 + cosines:
        magnitude = np.sqrt((mass - cos_sum)**2 + transverse_sq)
        curvature = (mass * cos_product - cos_sum) / (2 * magnitude**3)
        # Bin once per kx slice so we can integrate all Fermi energies at once.
        bins = np.searchsorted(positive_energies, 2 * magnitude, side="left")
        by_energy += np.bincount(bins.ravel(), curvature.ravel(), minlength=by_energy.size)

    # Converting from angular to Cartesian k and dividing the conductivity
    # by e^2/(2*h*a) gives -4*pi times the BZ mean of occupied curvature.
    ratios = -4 * np.pi * np.cumsum(by_energy[::-1])[::-1][1:] / grid_size**3
    # At E_F=0 the lower band is filled. The fixed-x slices have Chern number
    # -1 for m<2 and 0 for m>2, with a gap closing at x=±pi/2. Thus half the
    # slices contribute -1, giving a BZ mean of -1/(4*pi) and an exact ratio of 1.
    ratios[positive_energies == 0] = 1.0
    return ratios


def plot_comparison(energies, surface, sea, reference, output):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(energies, sea, color="tab:green", label="QuantumGeom Fermi sea")
    ax.plot(energies, reference, color="tab:orange", linestyle="--",
            label="Analytic Berry-curvature quadrature")
    ax.plot(energies, surface, color="tab:blue", marker="o", markersize=3,
            label="QuantumGeom Fermi surface")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set(xlabel=r"Fermi energy $E_F$ (eV)", ylabel=r"$\sigma_{yz}(E_F)/\sigma_{yz}(0)$",
           title="Hall conductivity of a two-band Weyl semimetal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=float, default=0.25, help="Energy step in eV; must divide 10")
    parser.add_argument("--points", type=int, default=2000, help="Initial Fermi-surface k-point count")
    parser.add_argument("--refinements", type=int, default=2, help="Fermi-surface mesh refinement rounds")
    parser.add_argument("--threads", type=int, default=2, help="OpenMP threads for QuantumGeom")
    parser.add_argument("--sea-points", type=int, default=20_000, help="Global Fermi-sea k-point count")
    parser.add_argument("--bz-grid", type=int, default=256, help="Analytic quadrature points per axis")
    parser.add_argument("--output", type=Path, default=Path("weyl_hall_conductivity.png"),
                        help="Figure path; the CSV uses the same stem")
    args = parser.parse_args()
    if not math.isfinite(args.step) or not 0 < args.step < BAND_EDGE_EV:
        parser.error("--step must be finite and between 0 and 10 eV")
    steps = round(BAND_EDGE_EV / args.step)
    if steps < 2 or not math.isclose(steps * args.step, BAND_EDGE_EV):
        parser.error("--step must divide 10 eV into at least two intervals")
    if min(args.points, args.threads, args.sea_points, args.bz_grid) < 1 or args.refinements < 0:
        parser.error("counts must be positive and refinement rounds cannot be negative")
    if args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf or .svg")

    positive_energies = np.linspace(0, BAND_EDGE_EV, steps + 1)
    energies = np.r_[-positive_energies[:0:-1], positive_energies]
    reference = occupied_state_ratios(positive_energies, args.bz_grid)
    reference = np.r_[reference[:0:-1], reference]  # The two-band answer is even in E_F.
    surface = np.full(energies.shape, np.nan)

    with TemporaryDirectory(prefix="quantumgeom-weyl-") as directory:
        # Bulk writes BrillouinZone.vtu in the working directory on construction.
        # Put that file and the diagnostic meshes in a temporary directory.
        cwd = Path.cwd()
        try:
            os.chdir(directory)
            bulk = qg.Bulk(str(MODEL))
        finally:
            os.chdir(cwd)
        mesh_file = Path(directory) / "fermi_surface.vtp"

        # The sea needs resolution throughout the occupied volume. Initialize
        # one global periodic mesh and reuse its tetrahedra for the whole scan.
        print(f"Integrating the Fermi sea on {args.sea_points:,} k-points", flush=True)
        bulk.initializeTheMeshParallel(args.sea_points, args.threads)
        sea_tensors = np.asarray(bulk.intrinsicMatrixOfFermiSeaIntegral(
            energies.tolist(), num_threads=args.threads
        ))
        sea = HALL_RATIO_SCALE * sea_tensors[:, 1, 2]

        for i, energy in enumerate(energies):
            # At the Weyl nodes and band extrema the level set contains only
            # points. Leave the surface sum undefined there instead of plotting zero.
            if energy in (-BAND_EDGE_EV, 0, BAND_EDGE_EV):
                continue
            print(f"[{i + 1}/{energies.size}] E_F = {energy:g} eV", flush=True)
            surface[i] = fermi_surface_ratio(
                bulk, energy, mesh_file, args.points, args.refinements, args.threads
            )

    plot_comparison(energies, surface, sea, reference, args.output)
    difference = surface - sea
    modulo_residual = difference - HALL_QUANTUM_RATIO * np.rint(difference / HALL_QUANTUM_RATIO)
    csv_file = args.output.with_suffix(".csv")
    np.savetxt(csv_file, np.column_stack((energies, surface, reference, sea, sea - reference,
                                        modulo_residual)), delimiter=",", comments="",
               header="Ef_eV,quantumgeom_surface_modulo_ratio,occupied_state_ratio,"
                      "quantumgeom_sea_ratio,sea_minus_reference_ratio,surface_minus_sea_modulo_ratio")
    print(f"Wrote {args.output} and {csv_file}")


if __name__ == "__main__":
    main()
