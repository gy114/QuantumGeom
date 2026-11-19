"""Small mesh initialization must survive progress reporting in worker threads."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLE_MODEL = Path(__file__).resolve().parents[1] / "examples/1band_wannier90_tb.dat"
INITIALIZE_MESH = """
import sys
try:
    import resource
except ImportError:
    pass
else:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import quantumGeom
bulk = quantumGeom.Bulk(sys.argv[1])
bulk.initializeTheMeshParallel(int(sys.argv[2]), int(sys.argv[3]))
"""


@pytest.mark.parametrize(
    "points, threads",
    [(0, 8), (1, 8), (39, 8), (40, 8), (41, 8), (1, 1), (4, 1), (5, 1)],
)
def test_small_mesh_progress(points, threads, tmp_path):
    # Run separately so a native crash becomes a test failure, not a pytest crash.
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_DYNAMIC="FALSE",
        OMP_THREAD_LIMIT=str(threads),
    )
    result = subprocess.run(
        [sys.executable, "-c", INITIALIZE_MESH, str(EXAMPLE_MODEL), str(points), str(threads)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Initializing {points} points on {threads} threads exited with "
        f"{result.returncode}.\n{result.stdout}\n{result.stderr}"
    )
    assert f"{points}/{points} vertices evaluated. All done." in result.stdout
