"""Chain must preserve the logical entries of NumPy hopping matrices."""

import os
import subprocess
import sys

import numpy as np
import pytest

import quantumGeom


def _with_layout(matrix, layout):
    if layout == "sliced":
        storage = np.full(
            (2 * matrix.shape[0], 2 * matrix.shape[1]),
            100 + 200j,
            dtype=np.complex128,
        )
        view = storage[::2, ::2]
        view[:] = matrix
        return view
    return np.array(matrix, dtype=np.complex128, order=layout)


@pytest.mark.parametrize("layout", ["C", "F", "sliced"])
def test_onsite_matrix_preserves_complex_entries(layout):
    onsite = np.array([[1, 2 + 3j], [2 - 3j, 4]], dtype=np.complex128)
    chain = quantumGeom.Chain([_with_layout(onsite, layout)], [0])

    np.testing.assert_array_equal(chain.generateLocalHamiltonian(0.0), onsite)


@pytest.mark.parametrize("layout", ["C", "F", "sliced"])
def test_hopping_matrices_preserve_bloch_hamiltonian(layout):
    onsite = np.array([[1, 2 + 3j], [2 - 3j, 4]], dtype=np.complex128)
    hopping = np.array(
        [[0.1 + 0.2j, 0.3 - 0.4j], [0.7 + 0.6j, -0.2 + 0.5j]],
        dtype=np.complex128,
    )
    reverse_hopping = hopping.conj().T
    chain = quantumGeom.Chain(
        [_with_layout(matrix, layout) for matrix in (onsite, hopping, reverse_hopping)],
        [0, 1, -1],
    )
    k = 0.137
    phase = np.exp(2j * np.pi * k)
    expected = onsite + phase * hopping + phase.conjugate() * reverse_hopping

    np.testing.assert_allclose(
        chain.generateLocalHamiltonian(k), expected, rtol=0, atol=1e-13
    )


def test_returned_hamiltonian_owns_its_memory():
    script = r"""
import numpy as np
import quantumGeom

onsite = np.array([[1, 2 + 3j], [2 - 3j, 4]], dtype=np.complex128)
chain = quantumGeom.Chain([onsite], [0])
hamiltonian = chain.generateLocalHamiltonian(0.0)

# Exercise the array after the temporary C++ return value has been destroyed.
assert hamiltonian.flags.owndata
for _ in range(100):
    np.empty((1024, 1024), dtype=np.complex128)
np.testing.assert_array_equal(hamiltonian, onsite)
print(hamiltonian)
"""
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Retaining a returned Hamiltonian exited with {result.returncode}.\n"
        f"{result.stdout}\n{result.stderr}"
    )
