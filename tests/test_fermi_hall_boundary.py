"""Check the seam-corrected Hall response against occupied-state integration."""

from pathlib import Path

import numpy as np
import pytest

import quantumGeom as qg
from test_fermi_wilson import _duplicate_weyl_model


WEYL_MODEL = Path(__file__).resolve().parents[1] / "examples/weyl_semi_tb.dat"


def _make_surface(model, energy, band):
    bulk = qg.Bulk(str(model))
    bulk.initializeTheMeshParallel(1800, 1)
    bulk.refineTheMeshForOneBandParallel(2, band, energy, 1)
    surface = qg.FermiSurface(energy, band)
    surface.generateSurface(bulk)
    return bulk, surface


def _hall_ratio(surface):
    tensor = surface.intrinsicMatrixOfHaldaneIntegral()
    # With a = 1 angstrom, the normalization is e^2/(2 h a).
    return -tensor[1, 2] / (2 * np.pi**2)


def _difference_modulo_hall_quantum(value, reference):
    # A reciprocal-lattice shift is two units in this normalization.
    return (value - reference + 1.0) % 2.0 - 1.0


@pytest.mark.parametrize(
    "energy, band, occupied_reference",
    [(-3.0, 0, 0.12718), (3.0, 1, 0.12718), (1.5, 1, 0.85649)],
)
def test_weyl_hall_boundary(energy, band, occupied_reference, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Bulk writes a Brillouin-zone file on construction.
    bulk, surface = _make_surface(WEYL_MODEL, energy, band)
    surface.obtainVertexEigenstatesParallel(bulk, 1)
    if energy == 3.0:
        with pytest.raises(RuntimeError, match="obtainBerryCurvature"):
            surface.intrinsicMatrixOfHaldaneIntegral()
    surface.obtainBerryCurvature()

    corrected = _hall_ratio(surface)
    # The occupied-state reference at ±3 eV includes the BZ-cut contribution.
    assert abs(_difference_modulo_hall_quantum(corrected, occupied_reference)) < 0.025

    if energy == 3.0:
        # The full two-band frame has trivial holonomy, but both the surface
        # and seam terms for this resolved sheet use the selected band.
        surface.obtainVertexEigenstatesParallel(bulk, 1, band_min=0, band_max=1)
        surface.obtainBerryCurvature()
        assert abs(_difference_modulo_hall_quantum(_hall_ratio(surface), corrected)) < 1e-10


def test_hall_integral_requires_boundary_correction():
    surface = qg.FermiSurface(3.0, 1)
    with pytest.raises(TypeError):
        surface.intrinsicMatrixOfHaldaneIntegral(include_boundary=False)
    with pytest.raises(TypeError):
        surface.intrinsicMatrixOfHaldaneIntegral(False)


def test_shared_weyl_doublet_uses_determinant_seam_links(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "doubled_weyl.dat"
    _duplicate_weyl_model(model)
    bulk, surface = _make_surface(model, 3.0, 2)

    # The constant block splitting picks one copy on the same geometry.
    bulk.setSpinSplittingEta(0.01)
    surface.obtainVertexEigenstatesParallel(bulk, 1)
    surface.obtainBerryCurvature()
    single_corrected = _hall_ratio(surface)

    bulk.setSpinSplittingEta(0.0)
    surface.obtainVertexEigenstatesParallel(bulk, 1, band_min=2, band_max=3)
    surface.obtainBerryCurvature()
    with pytest.raises(RuntimeError, match="requires Berry links"):
        _hall_ratio(surface)

    surface.obtainBerryCurvature(shared_boundary=True)
    doublet_corrected = _hall_ratio(surface)
    assert abs(_difference_modulo_hall_quantum(doublet_corrected, 2 * single_corrected)) < 1e-10
