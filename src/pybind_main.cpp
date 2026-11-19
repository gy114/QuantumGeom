/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */



#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "version.h"
#include "Bulk.hpp"
#include "FermiSurface.hpp"
#include "Chain.hpp"

namespace py = pybind11;

// Convert from numpy array to arma::vec
namespace pybind11 { namespace detail {
        template <> struct type_caster<arma::vec> {
        public:
            PYBIND11_TYPE_CASTER(arma::vec, _("numpy.ndarray"));

            // Conversion from Python to C++
            bool load(handle src, bool) {
                auto buf = py::array_t<double>::ensure(src);
                if (!buf) return false; // Conversion fails if the input is not a double array

                // Ensure the array is contiguous
                if (!buf.request().ptr) return false;

                // Use the buffer info to construct an arma::vec without copying
                auto info = buf.request();
                value = arma::vec(static_cast<double *>(info.ptr), info.size, false /* copy_aux_mem */, true /* strict */);
                return true;
            }

            // Conversion from C++ to Python
            static handle cast(const arma::vec& src, return_value_policy policy, handle parent) {
                // When returning to Python, make a copy of the data
                if (policy == return_value_policy::copy || policy == return_value_policy::automatic) {
                    return py::array_t<double>(src.n_elem, src.memptr()).release();
                }
                    // Or use the existing memory (requires the source data to stay alive)
                else {
                    return py::array_t<double>(
                            src.n_elem,  // Number of elements
                            src.memptr(),  // The data pointer
                            py::cast(src)  // Ensure the data stays alive
                    ).release();
                }
            }
        };
    }
} // namespace pybind11::detail
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Conversion from numpy array to arma::vec3
namespace pybind11 { namespace detail {
        template <> struct type_caster<arma::vec3> {
        public:
        PYBIND11_TYPE_CASTER(arma::vec3, _("numpy.ndarray"));

            // Conversion from Python to C++
            bool load(handle src, bool) {
                auto buf = py::array_t<double>::ensure(src);
                if (!buf || buf.size() != 3) return false; // Ensure it's a double array with size 3

                // Ensure the array is contiguous
                if (!buf.request().ptr) return false;

                // Copy data into arma::vec3
                auto info = buf.request();
                double* data_ptr = static_cast<double *>(info.ptr);
                value[0] = data_ptr[0];
                value[1] = data_ptr[1];
                value[2] = data_ptr[2];
                return true;
            }

            // Conversion from C++ to Python
            static handle cast(const arma::vec3& src, return_value_policy /* policy */, handle /* parent */) {
                // Correctly creating a NumPy array from arma::vec3
                return py::array_t<double>(
                        {3}, // shape of the array
                        {sizeof(double)}, // strides (distance in bytes between consecutive elements)
                        src.memptr(), // the data pointer
                        py::none() // parent, to keep alive if necessary
                );
            }
        };
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Conversion for arma::mat33
namespace pybind11 { namespace detail {

        template <> struct type_caster<arma::mat33> {
        public:
        PYBIND11_TYPE_CASTER(arma::mat33, _("numpy.ndarray"));

            // Conversion from Python to C++
            bool load(handle src, bool) {
                auto buf = py::array_t<double>::ensure(src);
                if (!buf) return false; // Conversion fails if the input is not a double array

                // Ensure the array is contiguous and has the correct dimensions
                if (buf.ndim() != 2 || buf.shape(0) != 3 || buf.shape(1) != 3) return false;

                auto info = buf.request();
                arma::mat33 tempMat; // Temporary matrix for data

                // Copy data into the arma::mat33
                double* data_ptr = static_cast<double*>(info.ptr);
                for (size_t col = 0; col < 3; ++col) {
                    for (size_t row = 0; row < 3; ++row) {
                        tempMat(row, col) = data_ptr[col * 3 + row]; // Column-major order
                    }
                }

                value = std::move(tempMat);
                return true;
            }

            // Conversion from C++ to Python
            static handle cast(const arma::mat33& src, return_value_policy /* policy */, handle /* parent */) {
                // Initialize a new NumPy array of shape (3, 3)
                auto result_array = py::array_t<double>({3, 3});

                // Obtain a mutable view of the result_array
                auto r = result_array.mutable_unchecked<2>(); // For a 2D array

                // Copy logical row/column entries; the accessors handle storage order.
                for (ssize_t i = 0; i < 3; ++i) {
                    for (ssize_t j = 0; j < 3; ++j) {
                        r(i, j) = src(i, j);
                    }
                }

                return result_array.release();
            }
        };
    }
} // namespace pybind11::detail

// Conversion for arma::cx_mat
namespace pybind11 { namespace detail {
        template <> struct type_caster<arma::cx_mat> {
        public:
        PYBIND11_TYPE_CASTER(arma::cx_mat, _("numpy.ndarray"));

            // Python -> C++
            bool load(handle src, bool) {
                auto buf = py::array_t<std::complex<double>, py::array::forcecast>::ensure(src);
                if (!buf) return false;
                auto info = buf.request();
                if (info.ndim != 2) return false;

                value.set_size(info.shape[0], info.shape[1]);
                auto input = buf.unchecked<2>();
                for (ssize_t row = 0; row < info.shape[0]; ++row) {
                    for (ssize_t col = 0; col < info.shape[1]; ++col) {
                        value(row, col) = input(row, col);
                    }
                }
                return true;
            }

            // C++ -> Python
            static handle cast(const arma::cx_mat& src, return_value_policy /* policy */, handle /* parent */) {
                auto result = py::array_t<std::complex<double>>(
                    {static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols)});
                auto output = result.mutable_unchecked<2>();
                for (ssize_t row = 0; row < static_cast<ssize_t>(src.n_rows); ++row) {
                    for (ssize_t col = 0; col < static_cast<ssize_t>(src.n_cols); ++col) {
                        output(row, col) = src(row, col);
                    }
                }
                return result.release();
            }
        };
    }
} // namespace pybind11::detail

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(quantumGeom, m) {
    m.attr("__version__") = QUANTUMGEOM_VERSION;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    py::class_<Bulk>(m, "Bulk", "Bulk", R"pbdoc(
    Bulk is a class describing a bulk tight-binding model given by Wannier90.

    Constructors:
        Bulk(a1, a2, a3, wannierHrDatFileName) - Initialize with three lattice vectors and a wannier90_hr.dat file.

        Bulk(wannierWinFileName, wannierHrDatFileName) - Initialize with two input files, one is wannier90.win, the other is wannier90_hr.dat.

        Bulk(wannierTbFileName) - Initialize with only one inputfile wannier90_tb.dat.

    Notes:
        Typically, wannier90.win contains the lattice vectors and the high-symmetry line to plot the tight-binding bands.
        Here, wannier90_hr.dat is a file using the inherited wannier90 format, whereas wannier90_tb.dat is a format given by later versions.
        Wannier90_hr.dat does not contain the lattice constants, and therefore a1, a2 and a3 need to be provided upon initialization.
        Wannier90_tb.dat contains the lattice vectors, so a1, a2 and a3 are not needed.
    Returns:
        A bulk object with the containers ready to be initialized.

)pbdoc")
// Binding for the first constructor
        .def(py::init([](const arma::vec& a1, const arma::vec& a2, const arma::vec& a3, const std::string& wannierHrDatFileName) {
            return new Bulk(const_cast<arma::vec&>(a1), const_cast<arma::vec&>(a2), const_cast<arma::vec&>(a3), wannierHrDatFileName.c_str());
        }), py::arg("a1"), py::arg("a2"), py::arg("a3"), py::arg("wannierHrDatFileName"))

// Binding for the second constructor
         //////////////////////////////////////////////////////////////////////////////////////////////////////
        .def(py::init([](const std::string& wannierWinFileName, const std::string& wannierHrDatFileName) {
                // Release the GIL before the potentially long-running operation
                py::gil_scoped_release release;

                // After releasing the GIL, create a new Bulk instance
                // This is where you would normally do any long initialization that doesn't
                // require interaction with Python objects.
                Bulk* bulk = new Bulk(wannierWinFileName.c_str(), wannierHrDatFileName.c_str());

                // The GIL is automatically reacquired when 'release' goes out of scope

                // Return the newly created instance
                return bulk;
            }), py::arg("wannierWinFileName"), py::arg("wannierHrDatFileName"))
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        .def(py::init([](const std::string& wannierTbFileName) {
                // Release the GIL before the potentially long-running operation
                py::gil_scoped_release release;

                // After releasing the GIL, create a new Bulk instance
                // This is where you would normally do any long initialization that doesn't
                // require interaction with Python objects.
                Bulk* bulk = new Bulk(wannierTbFileName.c_str());

                // The GIL is automatically reacquired when 'release' goes out of scope

                // Return the newly created instance
                return bulk;
            }), py::arg("wannierTbFileName"))
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        .def("outputBandStructureAlongHighSymmetry", [](Bulk &self, const std::string &inputFilename, const std::string &outputFilename, int numStepsFirstLine) {
                // Release the GIL before starting the long-running operation
                py::gil_scoped_release release;
                // Now the GIL is released, and the following operation can run
                // concurrently with other Python threads
                self.outputBandStructureAlongHighSymmetry(inputFilename.c_str(), outputFilename.c_str(), numStepsFirstLine);
                // The GIL is automatically reacquired when 'release' goes out of scope
                }, py::arg("inputFilename"), py::arg("outputFilename"), py::arg("numStepsFirstLine"),
             R"pbdoc(
                        Write the band structure of the tight-binding model along the high-symmetry line to a text file.

                        Parameters:
                            inputFilename:     a wannier win file that contains the high-symmetry lines, the lattice vectors and the tight-binding model.
                            outputFilename:    a string specifying the name of the output file.
                            numStepsFirstLine: an integer specifying the number of points to contain in the first interval of the high-symmetry line.
                        Returns:
                            None.
                        )pbdoc" )
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        .def("initializeTheMeshParallel", [](Bulk& self, int num_of_points, int num_OpenMP_threads) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Now that the GIL is released, call the function.
                     // No Python API interactions should occur within this function call.
                     self.initializeTheMeshParallel(num_of_points, num_OpenMP_threads);

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 }, py::arg("num_of_points"), py::arg("num_OpenMP_threads"),
                 R"pbdoc(
                        Initialize the periodic Delaunay mesh by randomly sampling points in the affine Brillouin zone (-0.5 -0.5 -0.5) -> (0.5 0.5 0.5)
                        Multiple OpenMP threads will be used to calculate the eigenvalues for each inserted vertex. In the cuboid we insert num_of_points
                        random points. The bulk mesh stores eigenvalues; eigenvectors are evaluated by the Berry routines as needed.

                        Parameters:
                            num_of_points: The number of points to initialize in the mesh.
                            num_OpenMP_threads: number of cores within a node that is requested to calculate the eigenvalues.
                        Returns:
                            None.
                        )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
                    //refineTheMeshForOneBandParallel(int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads);
            .def("refineTheMeshForOneBandParallel", [](Bulk& self, int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Execute the function with the GIL released
                     self.refineTheMeshForOneBandParallel(num_refine, iBand, Ef, num_OpenMP_threads);

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 }, py::arg("num_refine"), py::arg("iBand"), py::arg("Ef"), py::arg("num_OpenMP_threads"),
                 R"pbdoc(
                                Refine the mesh for the i-th energy band by interpolation. Only eigenvalues are stored for the new vertices.

                                Parameters:
                                        num_refine: the number of refinement rounds. Each round the points in the mesh near the Fermi surface
                                                    doubles, so usually we keep this number to be 3~5. The choice also depends on the initial number of points.
                                        iBand:     the band index that the refinement is trying to resolve.
                                        Ef:         the Fermi energy that the refinement is trying to resolve.
                                        num_OpenMP_threads: number of cores within a node that is requested to calculate the eigenvalues.

                                Returns:
                                        None.
                        )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("refineMeshParallelForBerryPhase", [](Bulk& self, int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads, int depth) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Execute the function with the GIL released
                     self.refineMeshParallelForBerryPhase(num_refine, iBand, Ef, num_OpenMP_threads, depth);

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 }, py::arg("num_refine"), py::arg("iBand"), py::arg("Ef"), py::arg("num_OpenMP_threads"), py::arg("depth"),
                 R"pbdoc(
                                This method refines the mesh by inserting points at the centroids of candidate cells with the largest Berry-curvature magnitude.
                                The selected band's eigenvectors are evaluated temporarily for this ranking.

                                Parameters:
                                        num_refine: the number of refinement rounds. Each round the points in the mesh near the Fermi surface
                                                    doubles, so usually we keep this number to be 3~5. The choice also depends on the initial number of points.
                                        iBand:     the band index that the refinement is trying to resolve.
                                        Ef:         the Fermi energy that the refinement is trying to resolve.
                                        num_OpenMP_threads: number of cores within a node that is requested to calculate the eigenvalues.
                                        depth:      percentage of candidate cells to refine, clamped to [0, 100].
                                                    The selected count is rounded down; 100 selects all candidate cells.

                                Returns:
                                        None.
                        )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("calculateBerryFluxBZ", [](Bulk& self) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Execute the function with the GIL released
                     self.calculateBerryFluxBZ();

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 },
                 R"pbdoc(
                                Calculate Berry flux around each bulk vertex and through the cell facets.
                                The eigenvectors used for this calculation are temporary. The flux can be exported to a VTU file.

                                Parameters:
                                        None.

                                Returns:
                                        None.
                        )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("intrinsicMatrixOfFermiSeaIntegral", [](Bulk& self, const std::vector<double>& energies, int threads) {
                     py::gil_scoped_release release;
                     return self.intrinsicMatrixOfFermiSeaIntegral(energies, threads);
                 }, py::arg("fermi_energies"), py::arg("num_threads") = 1,
                 R"pbdoc(
                        Integrate the occupied-state Berry curvature of the tight-binding model at zero temperature.
                        Initialize the periodic bulk mesh for the current Hamiltonian before calling. Band energies
                        interpolated inside each tetrahedron give its occupied fractions. The curvature is sampled
                        at its centroid and reused for all requested Fermi energies.

                        Parameters:
                            fermi_energies: One-dimensional sequence of Fermi energies in eV.
                            num_threads: Positive number of OpenMP threads.

                        Returns:
                            list[numpy.ndarray]: One antisymmetric 3x3 tensor per energy in input order, in inverse
                            angstroms. Multiply by -e^2 / (hbar * (2*pi)^3) in consistent units to get conductivity.
                            The tensor includes fully occupied bands and fixes the Hall quantum.
                 )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("findExtrema", &Bulk::findExtrema,
                py::arg("iBand"),
                 R"pbdoc(
                                This method search through the existing points in the mesh and writes the extrema values of the iBand-th band to the console.

                                Parameters:
                                        iBand:     the band index.
                                Returns:
                                        None.
                 )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("setSpinSplittingEta", &Bulk::setSmallSpinSplitting,
                 py::arg("eta"),
                 R"pbdoc(
                                Manually set the spin splitting energy.

                                Parameters:
                                    eta:  The spin splitting energy in eV.
                                Returns:
                                    None.
                 )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("writeMeshToVtu", [](Bulk& self, const std::string filename, arma::uword iBand) {
                // Release the GIL before starting the potentially long-running operation
                py::gil_scoped_release release;

                // Execute the function with the GIL released
                self.writeMeshToVtu(filename, iBand);

                // The GIL is automatically reacquired when 'release' goes out of scope
                }, py::arg("filename"), py::arg("iBand"),
                R"pbdoc(
                                This method writes the current 3D periodic Delaunay mesh to a vtu file.
                                With an extension of vtu, the file can be viewed and edited using Paraview.

                                Parameters:
                                        filename: the name of the file, usually it has vtu as its extension.
                                        iBand:    the band index of which the energies will be written as vertex data.
                                Returns:
                                        None.
                )pbdoc");

    ////////////////////////////////////// FermiSurface ////////////////////////////////////////////////////////////////
    py::class_<FermiSurface>(m, "FermiSurface",
                             R"pbdoc(
                                FermiSurface is a class describing a sheet of Fermi surface of a Bulk.
                                It contains an exact polygonal mesh describing the geometry.
                                The vertices can be filled with eigenstates to calculate Berry phase and intrinsic Hall conductivities.
                                The faces can be filled with eigenstates to calculate extrinsic (spin) Hall conductivity and magnetoresistance.

                                Constructors:
                                    FermiSurface(fermi_energy, band_id)

                                Parameters:
                                    fermi_energy: Fermi energy in eV.
                                    band_id: an integer specifying the band ID. The ID of the lowest band is 0.

                                Returns:
                                    A FermiSurface object with the containers ready to be initialized.
                            )pbdoc")
            .def(py::init<double, arma::uword>())
             ////////////////////////////////////////////////////////////////////////////////////////////
            .def("generateSurface", [](FermiSurface &self, Bulk& bulk) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Now that the GIL is released, call the function
                     self.generateSurface(bulk);

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 }, py::arg("Bulk"),
                 R"pbdoc(
                This method generate the Fermi surface by marching through the unique cells in the periodic Delaunay mesh.

                Parameters:
                    Bulk: a Bulk object is needed because it contains the refined periodic Delaunay mesh.

                Returns:
                    None.
            )pbdoc")
             //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("obtainVertexEigenstatesParallel", [](FermiSurface &self, Bulk& bulk, int num_threads,
                                                       std::optional<arma::uword> bandMin,
                                                       std::optional<arma::uword> bandMax,
                                                       std::optional<double> subspaceGap,
                                                       double degeneracyTolerance) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Execute the function with the GIL released
                     self.obtainVertexEigenstatesParallel(bulk, num_threads, bandMin, bandMax,
                                                           subspaceGap, degeneracyTolerance);

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 }, py::arg("Bulk"), py::arg("num_threads"), py::arg("band_min") = py::none(),
                    py::arg("band_max") = py::none(), py::arg("subspace_gap") = py::none(),
                    py::arg("degeneracy_tolerance") = 1e-8,
                 R"pbdoc(
                        Fill the vertices with eigenstates from one fixed band range for this surface.

                        Parameters:
                            Bulk: The Bulk object used to generate the surface.
                            num_threads: Number of OpenMP threads.
                            band_min, band_max: Inclusive zero-based band bounds containing the sheet band.
                                Omit both to use the sheet band alone or to select a range using subspace_gap.
                            subspace_gap: Expand the fixed range across adjacent gaps this small (in eV)
                                anywhere among the sampled vertices, including gaps involving newly added bands.
                            degeneracy_tolerance: Energy gap in eV below which band labels are treated as
                                unresolved. The selected range must include the entire unresolved block.
                )pbdoc")
             //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("obtainFaceEigenstatesAndSpinsParallel", [](FermiSurface &self, Bulk& bulk, int num_threads) {
                     // Release the GIL before starting the potentially long-running operation
                     py::gil_scoped_release release;

                     // Now that the GIL is released, call the function
                     self.obtainFaceEigenstatesAndSpinsParallel(bulk, num_threads);

                     // The GIL is automatically reacquired when 'release' goes out of scope
                 }, py::arg("Bulk"), py::arg("num_threads"),
                 R"pbdoc(
                        This method fills the faces (triangles) on the Fermi surface by the eigenstates for the k point at the centroids.
                        Spins are also stored.

                        Parameters:
                            Bulk: a Bulk object is needed because it contains the refined periodic Delaunay mesh.
                            num_threads: a number specifying the cores used by openMP. They must be on the same node.
                        Returns:
                            None.
                )pbdoc")

                    //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("obtainGroupVelocitiesParallel", [](FermiSurface &self, Bulk& bulk, int num_threads) {
                     py::gil_scoped_release release;
                     self.obtainGroupVelocitiesParallel(bulk, num_threads);
                 }, py::arg("Bulk"), py::arg("num_threads"),
                 R"pbdoc(
                        This method calculates the group velocities (dE/dk) for the faces (triangles) on the Fermi surface.
                        The unit is eV * angstrom

                        Parameters:
                            Bulk: a Bulk object is needed because it contains the refined periodic Delaunay mesh.
                            num_threads: a number specifying the cores used by openMP. They must be on the same node.
                        Returns:
                            None.
                 )pbdoc")
                    //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("obtainBerryCurvature", [](FermiSurface &self, bool sharedBoundary) {
                     py::gil_scoped_release release;
                     self.obtainBerryCurvatures(sharedBoundary);
                 }, py::arg("shared_boundary") = false,
                 R"pbdoc(
                        Compute triangle Wilson loops and the Berry curvature used for the Hall integral.

                        Resolved triangles use the sheet band's rank-one flux. A triangle with an
                        unresolved sheet band at a vertex reports that band's conventional share of the
                        multiplet flux in the diagnostics. Integrating the Hall response requires a
                        resolved band or a shared multiplet.

                        Set shared_boundary=True to integrate the total flux of a coincident multiplet
                        once. Use the lowest member of the fixed range as the sheet band. The sampled
                        energies must agree within degeneracy_tolerance and the states must occupy the
                        same side of the sheet.
                 )pbdoc")
            .def("wilsonTrianglePhases", [](const FermiSurface &self) {
                     const auto& phases = self.wilsonTrianglePhases();
                     const auto rows = static_cast<py::ssize_t>(phases.shares.n_rows);
                     const auto cols = static_cast<py::ssize_t>(phases.shares.n_cols);
                     py::array_t<double> shares({rows, cols});
                     py::array_t<bool> nearBranchCut(rows);
                     py::array_t<bool> hallUsesConvention(rows);
                     auto shareData = shares.mutable_unchecked<2>();
                     auto branchData = nearBranchCut.mutable_unchecked<1>();
                     auto conventionData = hallUsesConvention.mutable_unchecked<1>();
                     for (py::ssize_t row = 0; row < rows; ++row) {
                         branchData(row) = phases.nearBranchCut(row) != 0;
                         conventionData(row) = phases.hallUsesConvention(row) != 0;
                         for (py::ssize_t col = 0; col < cols; ++col) {
                             shareData(row, col) = phases.shares(row, col);
                         }
                     }
                     py::dict result;
                     result["bands"] = py::array_t<arma::uword>(cols, phases.bands.memptr());
                     result["total"] = py::array_t<double>(rows, phases.total.memptr());
                     result["shares"] = std::move(shares);
                     result["min_singular_value"] = py::array_t<double>(rows, phases.minSingularValue.memptr());
                     result["near_branch_cut"] = std::move(nearBranchCut);
                     result["hall_phase"] = py::array_t<double>(rows, phases.hallPhase.memptr());
                     result["hall_uses_convention"] = std::move(hallUsesConvention);
                     return result;
                 }, R"pbdoc(
                        Return arrays of triangle phases (radians) and diagnostics after obtainBerryCurvature.

                        ``bands`` lists the zero-based band indices for the columns of ``shares``.
                        Each row of ``shares`` sums to the multiplet's ``total`` modulo 2*pi. The shares
                        average the Wilson eigenphase weights at all three vertices, dividing equally
                        within unresolved blocks. ``min_singular_value`` measures the weakest
                        multiplet link. ``near_branch_cut`` flags branch ambiguity in the multiplet
                        or Hall phase. ``hall_phase`` gives the sheet phase, and ``hall_uses_convention``
                        marks conventional band shares that are only available as diagnostics.
                 )pbdoc")
                 //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("writeSurfaceVtp", [](FermiSurface &self, const std::string& filename, bool writeVelocity, bool writeSpin, bool writeBerryCurvature, bool binary) {
                     py::gil_scoped_release release;
                     self.writeSurfaceVtp(filename, writeVelocity, writeSpin, writeBerryCurvature, binary);
                 }, py::arg("filename"), py::arg("writeVelocity"), py::arg("writeSpin"), py::arg("writeBerryCurvature"), py::arg("binary"),
                 R"pbdoc(
                        This method writes the Fermi surface in the format of vtp. The information contained in the data is specified by the boolean switches.

                        Parameters:
                            filename: a string specifying the output file.
                            writeVelocity: a boolean variable that determines whether to write velocity vectors as cell (triangle) data.
                            writeSpin: the boolean switch for spin output.
                            writeBerryCurvature: the boolean switch for Berry curvature output.
                            binary: the boolean switch to determine if output as a binary file. If False, the format will be ASCII.
                        Returns:
                            None.
                 )pbdoc")
                 //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("writeBoundaryEdgesVtu", [](FermiSurface &self, const std::string& filename, bool binary) {
                     py::gil_scoped_release release;
                     self.writeBoundaryEdgesVtu(filename, binary);
                 }, py::arg("filename"), py::arg("binary"),
                 R"pbdoc(
                        This method writes the border of the surface mesh in the format of vtu.

                        Parameters:
                            filename: a string specifying the output file.
                            binary: the boolean switch to determine if output as a binary file. If False, the format will be ASCII.
                        Returns:
                            None.
                 )pbdoc")
                    //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("intrinsicMatrixOfHaldaneIntegral", [](const FermiSurface &self) {
                     py::gil_scoped_release release;
                     return self.intrinsicMatrixOfHaldaneIntegral();
                 },
                 R"pbdoc(
                        This method returns the Haldane integral tensor:

                        {0, -Iz,  Iy};
                        {Iz,  0, -Ix};
                        {-Iy, Ix,  0};

                        This matrix is in the unit of 1/angstrom.
                        The intrinsic conductivity tensor can be obtained by multiplying the tensor by -e^2/hbar/(2*pi)^3.
                        The integral uses the stored vertex coordinates and includes the Berry-phase
                        contribution from paired BZ seams. The Hall vector is defined modulo a
                        reciprocal-lattice vector.

                        Returns:
                            numpy.ndarray: The 3x3 Haldane integral tensor.
                        Raises:
                            RuntimeError: Berry curvature has not been calculated for a nonempty surface,
                                periodic border edges lack partners, or the band is unresolved. Use a
                                resolved band or shared_boundary=True for a coincident multiplet.
                 )pbdoc")
                    //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("velocityTensor", [](FermiSurface &self) {
                     py::gil_scoped_release release;
                     return self.velocityTensor();
                 },
                 R"pbdoc(
                        This method returns an averaged velocity tensor (3x3) (v) by integrating the following matrix:

                        {dE/dx * dE/dx,  dE/dx * dE/dy,  dE/dx * dE/dz};

                        {dE/dy * dE/dx,  dE/dy * dE/dy,  dE/dy * dE/dz};

                        {dE/dz * dE/dx,  dE/dz * dE/dy,  dE/dz * dE/dz};

                        This matrix is multiplied by 1/abs(dE/dx, dE/dy, dE/dz), then integrated throughout the Fermi surface.
                        The unit of returned (v) is eV / angstrom.

                        The current density can be calculated by C*(v)*(n)E, where:
                            - C = (1/2/pi)^3 * e^2 * tau / (hbar^2) is a constant,
                            - E is the magnitude of the electric field,
                            - e>0 is the electron charge,
                            - tau is a constant relaxation time,
                            - (n) is a 3x1 vector along the direction of the E field.

                        When calculating the Hall angle, C and E are canceled.

                        Parameters:
                            None.

                        Returns:
                            numpy.ndarray: A 3x3 matrix representing the averaged velocity tensor (v).
                 )pbdoc");

//////////////////////////////////////////// Chain: the 1D module ////////////////////////////////////

    py::class_<Chain>(m, "Chain", R"pbdoc(
    Initialize the Chain object from a Python list of complex NumPy matrices and a list of integers.

    Parameters
    ----------
    HR : list of np.ndarray
        List of 2D NumPy arrays with dtype=np.complex128.
        Example::

            HR = [
                numpy.array([[...], [...]], dtype=np.complex128),
                ...
            ]

    R : list of int
        List of integers corresponding to lattice displacements R.
        In the tight-binding model, the phase factor is
        :math:`\sum_R H(R) \exp(i k R)`
    )pbdoc")
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            .def(py::init([](py::list HR_py, std::vector<int> R) {
                     std::vector<arma::cx_mat> HR;
                     for (auto item : HR_py) {
                         // Match Armadillo's column-major storage when wrapping the buffer.
                         py::array_t<std::complex<double>, py::array::f_style | py::array::forcecast> arr =
                                 py::cast<py::array>(item);
                         auto buf = arr.request();
                         if (buf.ndim != 2)
                             throw std::runtime_error("Each H(R) must be a 2D complex array");

                         arma::cx_mat mat(reinterpret_cast<std::complex<double>*>(buf.ptr),
                                          buf.shape[0], buf.shape[1], false /* copy_aux_mem */);
                         HR.push_back(mat);
                     }
                     return new Chain(HR, R);
                 }),
                 py::arg("HR"), py::arg("R"),
                 R"pbdoc(
    Initialize the Chain object from a Python list of complex NumPy matrices and a list of integers.

    Parameters:
        HR: List of 2D NumPy arrays with dtype=np.complex128.
            HR should be defined as:
            HR = [
                        numpy.array([[...], [...]], dtype=np.complex128),
                        ...
            ]

        R: A list of integers corresponding to lattice displacements R.
                    In the tight-binding model, the phase factor is sum_R H(R) * exp(i * k * R)

)pbdoc")
//////////////////////////////////////////////////////////////////////////////////////////////////////
    .def("outputBandStructure", &Chain::outputBandStructure,
         py::arg("outputFilename"), py::arg("numSteps"),
         R"pbdoc(
            Write the band structure to a text file.

            Parameters:
                outputFilename: The path (and the name) of the output file.
                numSteps: Integer number of k-points sampled between -0.5 and 0.5.
         )pbdoc")

    .def("generateLocalHamiltonian", &Chain::generateLocalHamiltonian,
         py::arg("k1"),
         R"pbdoc(
            Return the Bloch Hamiltonian H(k) constructed from H(R) by doing the Bloch sum.

            Parameters:
                k1: Dimensionless momentum variable defined in [-0.5, 0.5).
         )pbdoc")
//////////////////////////////////////////////////////////////////////////////////////////////////////
    .def("findExtrema", &Chain::findExtrema,
         py::arg("iBand"),
         R"pbdoc(
            Print the extrema of the iBand-th band to stdout.

            Parameters:
                iBand: The band index counting from 0 (not 1).
         )pbdoc")
//////////////////////////////////////////////////////////////////////////////////////////////////////
    .def("zakPhase", [](Chain& self, arma::uword iBand, int N) {
             std::vector<double> zakResult;
             double phase = self.zakPhase(iBand, N, zakResult);
             return py::make_tuple(phase, zakResult);
         }, py::arg("iBand"), py::arg("N"),
         R"pbdoc(
            Compute the Zak phase for the iBand-th band using N samples. This function only works for the Hamiltonians denoted in orthogonal bases: S=Identity.

            Parameters:
                iBand: The band index upto which the integration is computed. iBand counts from 0, not 1.
                N: Number of k points in [-0.5, 0.5). For a nanohoop N is the number of repeating units.

            Returns:
                (zakPhase, phaseList): A tuple containing the Zak phase and the list of accumulated phases.
                Here the zakPhase is the net Zak phase integrated up to iBand-th band.
                The phaseList contains all the individual Zak phase for all the bands below iBand.
         )pbdoc")
//////////////////////////////////////////////////////////////////////////////////////////////////////

            .def("zakPhaseWithS", [](Chain& self, arma::uword iBand, int N) {
                     std::vector<double> zakResult;
                     double phase = self.zakPhaseWithS(iBand, N, zakResult);
                     return py::make_tuple(phase, zakResult);
                 }, py::arg("iBand"), py::arg("N"),
                 R"pbdoc(
            Compute the Zak phase for the iBand-th band using N samples,
            taking into account a non-orthogonal basis (an non-identity overlap matrix S).

            Parameters:
                iBand: The band index up to which the integration is computed. iBand counts from 0, not 1.
                N: Number of k points in [-0.5, 0.5). For a nanohoop N is the number of repeating units.

            Returns:
                (zakPhase, phaseList): A tuple containing the Zak phase and the list of accumulated phases
                computed with the non-orthogonal basis.
         )pbdoc")
//////////////////////////////////////////////////////////////////////////////////////////////////////
            .def("zakPhaseNonAbelian",
                 &Chain::zakPhaseNonAbelian,
                 py::arg("N"), py::arg("iBandMax"),
                 R"pbdoc(
            Compute the non-Abelian Zak phase for a set of bands. The overlap matrix S is used here.

            Parameters:
                N: Number of k points in [-0.5, 0.5). For a nanohoop N is the number of repeating units.
                iBandMax: Highest band index included in the non-Abelian subspace (counts from 0).

            Returns:
                The non-Abelian Zak phase as a scalar.
         )pbdoc")
//////////////////////////////////////////////////////////////////////////////////////////////////////
    .def("get_matrixN", &Chain::get_matrixN,
         R"pbdoc(
            Get the matrix dimension N of each H(R).

            Returns:
                Integer value of matrixN.
         )pbdoc");


}




