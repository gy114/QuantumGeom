.. _fermi-surfaces-and-berry-phases:

Fermi surfaces and Berry phases
===============================

Load a Wannier90 tight-binding model into :py:class:`~quantumGeom.Bulk`. To
extract the surface of a band at a given Fermi energy, initialize the periodic
Delaunay mesh and refine it near that energy. A :py:class:`~quantumGeom.FermiSurface`
contains the isosurface for one band, which can include several disconnected
pockets. Use zero-based band indices and Fermi energies in eV.

Run this example from the repository root.

.. code-block:: python

   import quantumGeom as qg

   bulk = qg.Bulk("examples/weyl_semi_tb.dat")
   ef, band, threads = -1.0, 0, 4
   bulk.initializeTheMeshParallel(2000, threads)
   bulk.refineTheMeshForOneBandParallel(3, band, ef, threads)

   surface = qg.FermiSurface(ef, band)
   surface.generateSurface(bulk)
   surface.obtainVertexEigenstatesParallel(bulk, threads)
   surface.obtainBerryCurvature()
   phases = surface.wilsonTrianglePhases()

:py:meth:`~quantumGeom.FermiSurface.obtainBerryCurvature` computes the phase
around each triangle from the vertex eigenstates.
:py:meth:`~quantumGeom.FermiSurface.wilsonTrianglePhases` returns the phases in
radians and diagnostics for checking the mesh and band resolution.

.. _band-groups-and-degeneracies:

Band groups and degeneracies
----------------------------

After generating a Fermi surface, pass an inclusive band range containing the
sheet band to :py:meth:`~quantumGeom.FermiSurface.obtainVertexEigenstatesParallel`.
The same band range is used at every vertex, including across periodic seams.
To select the range from the sampled energy gaps, use ``subspace_gap`` in eV.
For a model with a sheet in band 5 or 6, the explicit range looks like this.

.. code-block:: python

   surface.obtainVertexEigenstatesParallel(bulk, 4, band_min=5, band_max=6)
   surface.obtainBerryCurvature()
   phases = surface.wilsonTrianglePhases()

``phases["total"]`` is the determinant phase for each triangle.
``phases["shares"]`` distributes that phase among the bands by their overlaps
with the Wilson eigenvectors. The shares sum to the total modulo :math:`2\pi`.
The Hall calculation uses the rank-one phase on a resolved sheet. The flag
``phases["hall_uses_convention"]`` identifies triangles whose conventional shares
are available for inspection. The Hall integral requires a resolved sheet or
shared multiplet. Check mesh convergence if ``phases["near_branch_cut"]`` is set.

For a degenerate multiplet that shares a Fermi surface and the same occupied
side, generate the surface for its lowest band and call
``obtainBerryCurvature(shared_boundary=True)``. This integrates the full
multiplet phase once. Integrate the other sheets separately if they have
distinct occupation boundaries.
