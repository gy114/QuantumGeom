"""Exercise fixed-rank Wilson frames through the Fermi-surface Hall API."""

from pathlib import Path

import numpy as np
import pytest

import quantumGeom as qg


WEYL_MODEL = Path(__file__).resolve().parents[1] / "examples/weyl_semi_tb.dat"
ENERGY = -1.5


def _make_surface(bulk, energy, band):
    bulk.initializeTheMeshParallel(1000, 1)
    bulk.refineTheMeshForOneBandParallel(2, band, energy, 1)
    surface = qg.FermiSurface(energy, band)
    surface.generateSurface(bulk)
    return surface


def _duplicate_weyl_model(path):
    """Write H_Weyl(k) ⊕ H_Weyl(k), with exactly degenerate copies."""
    header, *blocks = WEYL_MODEL.read_text(encoding="utf-8").strip().split("\n\n")
    header = header.splitlines()
    header[4] = "4"
    output = ["\n".join(header)]
    for block in blocks:
        translation, *entries = block.splitlines()
        matrix = {}
        for entry in entries:
            row, column, real, imag = entry.split()
            matrix[int(row) - 1, int(column) - 1] = (real, imag)
        expanded = [translation]
        for row in range(4):
            for column in range(4):
                real, imag = (
                    matrix[row % 2, column % 2] if row // 2 == column // 2 else ("0", "0")
                )
                expanded.append(f"{row + 1} {column + 1} {real} {imag}")
        output.append("\n".join(expanded))
    path.write_text("\n\n".join(output) + "\n", encoding="utf-8")


@pytest.mark.parametrize("energy, band", [(ENERGY, 0), (-ENERGY, 1)])
def test_full_weyl_subspace_preserves_resolved_sheet_hall(energy, band, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Bulk writes its Brillouin-zone file on construction.
    bulk = qg.Bulk(str(WEYL_MODEL))
    surface = _make_surface(bulk, energy, band)

    surface.obtainVertexEigenstatesParallel(bulk, 1)
    surface.obtainBerryCurvature()
    single_band_tensor = surface.intrinsicMatrixOfHaldaneIntegral().copy()
    assert single_band_tensor[1, 2] > 10.0

    surface.obtainVertexEigenstatesParallel(bulk, 1, band_min=0, band_max=1)
    surface.obtainBerryCurvature()
    phases = surface.wilsonTrianglePhases()

    # A full-space loop has unit holonomy, while the resolved Weyl pocket has flux.
    assert phases["total"].size > 0
    np.testing.assert_array_equal(phases["bands"], [0, 1])
    assert phases["shares"].shape == (phases["total"].size, 2)
    np.testing.assert_allclose(phases["total"], 0.0, atol=1e-12)
    np.testing.assert_allclose(phases["shares"], 0.0, atol=1e-12)
    np.testing.assert_allclose(phases["min_singular_value"], 1.0, rtol=0.0, atol=1e-12)
    assert not np.any(phases["near_branch_cut"])
    np.testing.assert_allclose(
        surface.intrinsicMatrixOfHaldaneIntegral(), single_band_tensor, rtol=0.0, atol=1e-12
    )

    with pytest.raises(ValueError):
        surface.obtainBerryCurvature(shared_boundary=True)


def test_shared_doublet_hall_counts_trace_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "doubled_weyl.dat"
    _duplicate_weyl_model(model)
    bulk = qg.Bulk(str(model))
    surface = _make_surface(bulk, ENERGY, 0)

    # The constant splitting selects one copy without changing its eigenvectors.
    # Keep the mesh fixed so both calculations use the same geometric quadrature.
    bulk.setSpinSplittingEta(0.01)
    surface.obtainVertexEigenstatesParallel(bulk, 1)
    surface.obtainBerryCurvature()
    single_copy_tensor = surface.intrinsicMatrixOfHaldaneIntegral().copy()
    assert single_copy_tensor[1, 2] > 10.0

    bulk.setSpinSplittingEta(0.0)
    surface.obtainVertexEigenstatesParallel(bulk, 1, subspace_gap=0.05)
    surface.obtainBerryCurvature()
    convention = surface.wilsonTrianglePhases()
    assert convention["hall_uses_convention"].size > 0
    assert np.all(convention["hall_uses_convention"])
    np.testing.assert_allclose(convention["hall_phase"], convention["shares"][:, 0])
    with pytest.raises(RuntimeError, match="requires Berry links"):
        surface.intrinsicMatrixOfHaldaneIntegral()

    surface.obtainBerryCurvature(shared_boundary=True)
    phases = surface.wilsonTrianglePhases()

    np.testing.assert_array_equal(phases["bands"], [0, 1])
    assert not np.any(phases["hall_uses_convention"])
    np.testing.assert_allclose(phases["shares"][:, 0], phases["total"] / 2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(phases["shares"][:, 1], phases["total"] / 2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        surface.intrinsicMatrixOfHaldaneIntegral(), 2 * single_copy_tensor, rtol=0.0, atol=1e-10
    )

    # The upper member of the pair would integrate the same occupation boundary twice.
    second_member = qg.FermiSurface(ENERGY, 1)
    second_member.generateSurface(bulk)
    second_member.obtainVertexEigenstatesParallel(bulk, 1, subspace_gap=0.05)
    with pytest.raises(ValueError):
        second_member.obtainBerryCurvature(shared_boundary=True)
