"""Compare the native periodic-mesh Fermi-sea integral with Weyl references."""

from pathlib import Path

import numpy as np
import pytest

import quantumGeom as qg


WEYL_MODEL = Path(__file__).resolve().parents[1] / "examples/weyl_semi_tb.dat"


def test_fermi_sea_requires_valid_input_and_a_mesh(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bulk = qg.Bulk(str(WEYL_MODEL))
    assert bulk.intrinsicMatrixOfFermiSeaIntegral([]) == []

    with pytest.raises(ValueError):
        bulk.intrinsicMatrixOfFermiSeaIntegral([0.0], num_threads=0)
    with pytest.raises(ValueError):
        bulk.intrinsicMatrixOfFermiSeaIntegral([np.nan])
    with pytest.raises(RuntimeError):
        bulk.intrinsicMatrixOfFermiSeaIntegral([0.0])


def test_weyl_fermi_sea_hall_tensor(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Bulk writes its Brillouin-zone file on construction.
    bulk = qg.Bulk(str(WEYL_MODEL))
    bulk.initializeTheMeshParallel(6000, 2)

    # The bands run from -10 to +10 eV. An unsorted batch exercises the
    # return order, including a repeated energy and both occupation limits.
    energies = [3.0, -11.0, 0.0, -1.5, 11.0, -3.0, 1.5, 3.0]
    tensors = np.asarray(bulk.intrinsicMatrixOfFermiSeaIntegral(energies, num_threads=2))
    assert tensors.shape == (len(energies), 3, 3)
    assert np.all(np.isfinite(tensors))
    np.testing.assert_allclose(tensors, -tensors.swapaxes(1, 2), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(tensors[[1, 4]], 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(tensors[0], tensors[7], rtol=0.0, atol=1e-12)

    # The tensors use the Fermi-surface convention. For a = 1 Å the
    # dimensionless ratio to e^2/(2 h a) is -M_yz/(2 pi^2).
    hall_ratios = -tensors[:, 1, 2] / (2 * np.pi**2)
    # The node value is exact, with a looser mesh tolerance for the singular
    # curvature. The other references use independent analytic-curvature
    # quadrature. Absolute comparisons also check the Hall branch.
    assert hall_ratios[2] == pytest.approx(1.0, abs=0.2)
    np.testing.assert_allclose(hall_ratios[[3, 0]], [0.85649, 0.12718], rtol=0.0, atol=0.025)
    np.testing.assert_allclose(tensors[[0, 3]], tensors[[5, 6]], rtol=0.0, atol=1e-10)


def test_reflected_weyl_sea_matches_surface(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "reflected_weyl.dat"
    lines = WEYL_MODEL.read_text(encoding="utf-8").splitlines()
    lines[1] = "-1.00    0.00    0.00"
    model.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bulk = qg.Bulk(str(model))
    bulk.initializeTheMeshParallel(1200, 2)
    energy, band = 3.0, 1
    bulk.refineTheMeshForOneBandParallel(1, band, energy, 2)
    sea = bulk.intrinsicMatrixOfFermiSeaIntegral([energy], num_threads=2)[0]

    surface = qg.FermiSurface(energy, band)
    surface.generateSurface(bulk)
    surface.obtainVertexEigenstatesParallel(bulk, 2)
    surface.obtainBerryCurvature()
    hall_surface = surface.intrinsicMatrixOfHaldaneIntegral()

    # Reflection reverses the cell handedness without changing the analytic
    # yz conductivity of this model, whose x dependence is through cos(x).
    sea_ratio = -sea[1, 2] / (2 * np.pi**2)
    surface_ratio = -hall_surface[1, 2] / (2 * np.pi**2)
    assert sea_ratio == pytest.approx(0.12718, abs=0.025)
    assert abs((surface_ratio - sea_ratio + 1.0) % 2.0 - 1.0) < 0.025
