/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#include "FermiSurface.hpp"
#include "Bulk.hpp"
#include "TriangleWilsonLoop.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <omp.h>
#include "WriteToVTK.hpp"
#include <atomic>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>
#include <vtkCellArray.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <unordered_map> // Include for the mapping
#include <vtkLine.h> // Include for vtkLine
#include <vtkUnstructuredGrid.h> // Include for vtkUnstructuredGrid
#include <vtkXMLUnstructuredGridWriter.h> // Correct include for vtkXMLUnstructuredGridWriter



/*K::Point_3 to_inexact(const K_exact::Point_3& exact_point) {
    double x = CGAL::to_double(exact_point.x());
    double y = CGAL::to_double(exact_point.y());
    double z = CGAL::to_double(exact_point.z());
    return K::Point_3(x, y, z);
}*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FermiSurface::FermiSurface(double ef, arma::uword new_iBand)  {
    this->iBand = new_iBand;
    this->Ef = K_exact::FT(ef);
    this->header = {new_iBand, new_iBand};
    std::cout<< "Fermi surface of band " << iBand << " at Ef = " << ef << " is created." << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::prepareContainers() {
    this->groupVelocities123.clear();
    this->faceEigenstates.clear();
    this->listOfVertexPhysics.clear();
    this->hasSharedBoundary = false;
    this->hallUsesSharedBoundary = false;
    this->trianglePhases = {};
    this->triangleAreasXYZ.clear();
    this->spins.clear();
    this->berryCurvature.clear();
    this->berryFlux.clear();
    std::cout << "Containers on the Fermi surface of band " << this->iBand << " are ready." << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::generateSurface(Bulk& bulk) {
    if (bulk.theMesh.number_of_vertices() == 0)  {
        std::cout << "Err: the 3D mesh is empty. Need to initialize the Bulk object. " << std::endl;
        return;
    }
    if (this->iBand >= bulk.get_matrixN()) {
        throw std::invalid_argument("Fermi-surface band index is out of range.");
    }
    std::cout << "======== Starting to generate the Fermi surface of band " << this->iBand << " ========" << std::endl;
    this->theSurface = extractOneFermiSurfaceMesh(bulk, this->iBand, this->Ef);
    Iso_cuboid newDomain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    this->domain = newDomain;
    this->B = bulk.get_B();
    this->b1 = bulk.get_b1();
    this->b2 = bulk.get_b2();
    this->b3 = bulk.get_b3();
    //Surface_mesh sm;
    //CGAL::copy_face_graph(sm, theSurface);
    //CGAL::Polygon_mesh_processing::isotropic_remeshing(faces(sm), sm, )

    // set the ID:
    unsigned long long setID = 0;
    for (auto vi = this->theSurface.vertices_begin(); vi != this->theSurface.vertices_end(); ++vi)  {
        vi->id() = setID++;
    }
    setID = 0;
    for (auto fi = this->theSurface.faces_begin(); fi != this->theSurface.faces_end(); ++fi)  {
        fi->id() = setID++;
    }
    std::cout << "Surface created for band " << this->iBand << ". IDs of faces and vertices are set" << std::endl;
    std::cout << num_vertices(this->theSurface) << " vertices and " << num_faces(this->theSurface) << " faces are identified." << std::endl;
    std::cout << "The surface mesh has " << num_halfedges(theSurface) << " half edges and " << num_edges(theSurface) << " edges." << std::endl;
    std::cout << this->theSurface.size_of_border_edges() << " border edges are found." << std::endl;
    this->periodicConnectivity = buildPeriodicSurfaceConnectivity(this->theSurface);
    if (this->periodicConnectivity.unmatchedBorderHalfedges != 0) {
        std::cerr << "Warning: " << this->periodicConnectivity.unmatchedBorderHalfedges
                  << " Fermi-surface border edges lack a unique periodic partner.\n";
    }
    this->prepareContainers();
    this->calculateEuclideanAreaForAllTriangles();

    std::cout << "======================== the Fermi surface of band " << this->iBand << " generated ========\n\n\n" << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::obtainVertexEigenstatesParallel(Bulk& bulk, int num_threads,
                                                   std::optional<arma::uword> bandMin,
                                                   std::optional<arma::uword> bandMax,
                                                   std::optional<double> subspaceGap,
                                                   double degeneracyTolerance) {
    const auto bandCount = bulk.get_matrixN();
    if (num_threads < 1 || this->iBand >= bandCount) {
        throw std::invalid_argument("Expected a positive thread count and a valid sheet band index.");
    }
    if (!std::isfinite(degeneracyTolerance) || degeneracyTolerance < 0 ||
        (subspaceGap && (!std::isfinite(*subspaceGap) || *subspaceGap < 0))) {
        throw std::invalid_argument("Energy tolerances must be finite and nonnegative.");
    }
    if (bandMin.has_value() != bandMax.has_value() || (subspaceGap && bandMin)) {
        throw std::invalid_argument("Provide both band bounds or an automatic subspace gap.");
    }
    SheetSubspaceHeader bands{bandMin.value_or(this->iBand), bandMax.value_or(this->iBand)};
    if (bands.band_min > this->iBand || bands.band_max < this->iBand || bands.band_max >= bandCount) {
        throw std::invalid_argument("The band range must contain the sheet band and lie within the model spectrum.");
    }

    const auto& representatives = this->periodicConnectivity.vertexRepresentative;
    std::vector<arma::vec3> points(this->theSurface.size_of_vertices());
    for (auto vertex = this->theSurface.vertices_begin(); vertex != this->theSurface.vertices_end(); ++vertex) {
        if (representatives[vertex->id()] == vertex->id()) {
            // The tight-binding Hamiltonian is periodic in reduced coordinates. Use the
            // same frame for all copies so a link agrees on both sides of a seam.
            const auto p = map_to_exact_cuboid(vertex->point());
            points[vertex->id()] = {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
        }
    }
    if (subspaceGap) {
        bands = establishSubspace(bulk, points, std::max(*subspaceGap, degeneracyTolerance), num_threads);
    }

    std::cout << "Evaluating vertex eigenstates for surface band " << this->iBand
              << " in band range [" << bands.band_min << ", " << bands.band_max << "]..." << std::endl;
    std::vector<FermiVertexPhysics> physics(points.size());
    int failed = 0;
    double exteriorGap = std::numeric_limits<double>::infinity();
    #pragma omp parallel for num_threads(num_threads) reduction(|:failed) reduction(min:exteriorGap)
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (representatives[i] != i) {
            continue;
        }
        const auto& k = points[i];
        const auto hamiltonian = bulk.generateLocalHamiltonian(k(0), k(1), k(2));
        arma::vec energies;
        arma::cx_mat eigenvectors;
        if (!hamiltonian.is_finite() || !arma::eig_sym(energies, eigenvectors, hamiltonian)) {
            failed = 1;
            continue;
        }
        if (bands.band_min > 0) {
            exteriorGap = std::min(exteriorGap, energies(bands.band_min) - energies(bands.band_min - 1));
        }
        if (bands.band_max + 1 < bandCount) {
            exteriorGap = std::min(exteriorGap, energies(bands.band_max + 1) - energies(bands.band_max));
        }
        physics[i].energies = energies.subvec(bands.band_min, bands.band_max);
        physics[i].eigenvectors = eigenvectors.cols(bands.band_min, bands.band_max);
    }
    if (failed) {
        throw std::runtime_error("Could not diagonalize a Fermi-surface vertex Hamiltonian.");
    }
    if (exteriorGap <= degeneracyTolerance) {
        throw std::invalid_argument("The selected band range touches an excluded band at a surface vertex. "
                                    "Widen the fixed band range or use subspace_gap.");
    }
    const bool shared = bands.M() > 1 && !points.empty() &&
                        checkSharedBoundary(bulk, physics, bands, degeneracyTolerance);
    this->header = bands;
    this->listOfVertexPhysics = std::move(physics);
    this->degeneracyTolerance = degeneracyTolerance;
    this->hasSharedBoundary = shared;
    this->hallUsesSharedBoundary = false;
    this->berryCurvature.clear();
    this->berryFlux.clear();
    this->trianglePhases = {};
    std::cout << "Fermi-surface vertex eigenstates ready." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::obtainFaceEigenstatesAndSpinsParallel(Bulk& bulk, int num_threads)   {
    this->faceEigenstates.clear();
    for (unsigned long long i=0; i<num_faces(this->theSurface); i++)  {
        arma::cx_vec initialVec(bulk.get_matrixN(), arma::fill::zeros);
        this->faceEigenstates.push_back(initialVec);
    }
    unsigned long long total = num_faces(this->theSurface);
    // First record all the coordinates of centroids.
    std::vector<arma::vec3> listOfCentroids;
    listOfCentroids.resize(this->faceEigenstates.size());
    std::cout << "======== Starting to fill eigenstates on "<< total << " faces on the Fermi surface of band " << this->iBand <<" =========" << std::endl;
    for(auto fi = this->theSurface.faces_begin(); fi != this->theSurface.faces_end(); ++fi) {
        Polyhedron::Halfedge_around_facet_circulator hfc = fi->facet_begin();
        K_exact::Point_3 p1 = hfc->vertex()->point();
        ++hfc;
        K_exact::Point_3 p2 = hfc->vertex()->point();
        ++hfc;
        K_exact::Point_3 p3 = hfc->vertex()->point();
        K_exact::Point_3 centroid = CGAL::centroid(p1, p2, p3);
        //std::cout << "debug three points obtained.\n";
        listOfCentroids[fi->id()] = arma::vec3{CGAL::to_double(centroid[0]), CGAL::to_double(centroid[1]), CGAL::to_double(centroid[2])};
    }
    std::cout << num_faces(this->theSurface) << " centroids recorded to be parallely processed." << std::endl;

    omp_set_num_threads(num_threads);
    unsigned long long completedIterations = 0;
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of OpenMP threads : " << omp_get_num_threads() << std::endl;

        unsigned long long localCounter = 0; // thread-local counter
        const auto reportInterval = std::max(1ULL, total / omp_get_num_threads() / 5);

        #pragma omp for nowait
        for (unsigned long long i = 0; i < total; i++) {
            arma::vec3 k123 = listOfCentroids[i];
            arma::cx_mat H = bulk.generateLocalHamiltonian(k123[0], k123[1], k123[2]);
            arma::vec eigval;
            arma::cx_mat eigvec;
            arma::eig_sym(eigval, eigvec, H);

            // Ensure iBand is within the range [0, eigvec.n_cols-1]
            if (this->iBand < eigvec.n_cols) {
                this->faceEigenstates[i] = eigvec.col(this->iBand);
            } else {
                std::cerr << "iBand index out of range." << " face ID = " << i << std::endl;
                // Handle the error appropriately
            }
            localCounter++;
            if (localCounter % reportInterval == 0) {
                #pragma omp atomic
                completedIterations += localCounter;
                localCounter = 0; //Reset local counter after updating global counter.
                if (omp_get_thread_num() == 0)  {
                    #pragma omp critical
                    std::cout << completedIterations << "/" << total << " faces evaluated..." << std::endl;
                }
            }
        }
        if (localCounter > 0) {
            #pragma omp atomic
            completedIterations += localCounter;
        }
    }
    std::cout << completedIterations << "/" << total << " faces evaluated... all done." << std::endl;
    if (bulk.get_matrixN()%2 == 0) {
        this->calculateFaceSpins();
        std::cout << "======== Eigenstates and spins filled on " << total << " faces on the Fermi surface of band "
                  << this->iBand << " =========\n\n\n" << std::endl;
    }   else    {
        std::cout << "======== H is not spin resolved. Eigenstates (not spins) filled on " << total << " faces for band "
                << this->iBand << " =========\n\n\n" << std::endl;
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double find_shortest_edge_squared(const Polyhedron& polyhedron) {
    double min_edge_length_squared = std::numeric_limits<double>::max();

    // Iterate over all edges
    for (Polyhedron::Edge_const_iterator ei = polyhedron.edges_begin(); ei != polyhedron.edges_end(); ++ei) {
        // Get the vertices of the edge
        K_exact::Point_3 p1 = ei->vertex()->point();
        K_exact::Point_3 p2 = ei->opposite()->vertex()->point();

        // Calculate squared distance between the vertices
        double edge_length_squared = CGAL::to_double(CGAL::squared_distance(p1, p2));

        // Update minimum if this edge is shorter
        if (edge_length_squared < min_edge_length_squared) {
            min_edge_length_squared = edge_length_squared;
        }
    }

    return CGAL::to_double(min_edge_length_squared);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::obtainGroupVelocitiesParallelNewton(Bulk& bulk, double eta, int num_threads)  {
    // First, find the shortest edge on the surface.
    double shortestEdge = std::sqrt(find_shortest_edge_squared(this->theSurface));
    double step = shortestEdge * eta;
    // just in case the edges are too short.
    if (step < 1.0e-10)
        step = 1.0e-10;
    this->groupVelocities123.clear();
    this->groupVelocities123.resize(num_faces(this->theSurface));
    unsigned long long total = num_faces(this->theSurface);
    // First record all the coordinates of centroids.
    std::vector<arma::vec3> listOfCentroids;
    listOfCentroids.resize(this->groupVelocities123.size());
    std::cout << "======== Starting to fill velocities on "<< total << " faces on the Fermi surface of band " << this->iBand <<" =========" << std::endl;
    std::cout << "Shortest edge length is " << shortestEdge << std::endl;
    for(auto fi = this->theSurface.faces_begin(); fi != this->theSurface.faces_end(); ++fi) {
        Polyhedron::Halfedge_around_facet_circulator hfc = fi->facet_begin();
        K_exact::Point_3 p1 = hfc->vertex()->point();
        ++hfc;
        K_exact::Point_3 p2 = hfc->vertex()->point();
        ++hfc;
        K_exact::Point_3 p3 = hfc->vertex()->point();
        K_exact::Point_3 centroid = CGAL::centroid(p1, p2, p3);
        //std::cout << "debug three points obtained.\n";
        listOfCentroids[fi->id()] = arma::vec3{CGAL::to_double(centroid[0]), CGAL::to_double(centroid[1]), CGAL::to_double(centroid[2])};
    }
    std::cout << num_faces(this->theSurface) << " centroids recorded to be parallelly processed." << std::endl;
    // Now, we obtain the group velocities using openMP:

    // First, create an atomic object to track the progress.
    unsigned long long completedIterations = 0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of OpenMP threads : " << omp_get_num_threads() << std::endl;

        unsigned long long localCounter = 0; // thread-local counter
        const auto reportInterval = std::max(1ULL, total / omp_get_num_threads() / 5);
        arma::cx_mat H(bulk.get_matrixN(), bulk.get_matrixN()); // local matrix containers
        arma::cx_mat HdX(bulk.get_matrixN(), bulk.get_matrixN());
        arma::cx_mat HdY(bulk.get_matrixN(), bulk.get_matrixN());
        arma::cx_mat HdZ(bulk.get_matrixN(), bulk.get_matrixN());
        double E, Edx, Edy, Edz; // local energy value containers
        arma::vec eigval0(bulk.get_matrixN()); // local eigenvalue containers.
        arma::vec eigvaldx(bulk.get_matrixN());
        arma::vec eigvaldy(bulk.get_matrixN());
        arma::vec eigvaldz(bulk.get_matrixN());

        #pragma omp for nowait
        for (unsigned long long i = 0; i < total; i++) {

            arma::vec3 k123 = listOfCentroids[i];

        // Generate Hamiltonian and calculate eigenvalues for the original point
            H = bulk.generateLocalHamiltonian(k123(0), k123(1), k123[2]);
            eigval0 = arma::eig_sym(H);
            E = eigval0(this->iBand);

        // Calculate eigenvalues for the point shifted by 'step' in the x-direction
            HdX = bulk.generateLocalHamiltonian(k123[0] + step, k123[1], k123[2]);
            eigvaldx = arma::eig_sym(HdX);
            Edx = eigvaldx(this->iBand);

        // Calculate eigenvalues for the point shifted by 'step' in the y-direction
            HdY = bulk.generateLocalHamiltonian(k123[0], k123[1] + step, k123[2]);
            eigvaldy = arma::eig_sym(HdY);
            Edy = eigvaldy(this->iBand);

        // Calculate eigenvalues for the point shifted by 'step' in the z-direction
            HdZ = bulk.generateLocalHamiltonian(k123[0], k123[1], k123[2] + step);
            eigvaldz = arma::eig_sym(HdZ);
            Edz = eigvaldz(this->iBand);

        // Store the calculated group velocities
            this->groupVelocities123[i] = arma::vec3{(Edx - E) / step, (Edy - E) / step, (Edz - E) / step};
            localCounter++;
            if (localCounter % reportInterval == 0) {
                #pragma omp atomic
                completedIterations += localCounter;
                localCounter = 0; //Reset local counter after updating global counter.
                if (omp_get_thread_num() == 0) {
                    #pragma omp critical
                    std::cout << completedIterations << "/" << total << " velocities evaluated..." << std::endl;
                }
            }
        }
        if (localCounter > 0) {
            #pragma omp atomic
            completedIterations += localCounter;
        }
    }
    std::cout << completedIterations << "/" << total << " velocities evaluated... all done." << std::endl;
    std::cout << "======== Velocities filled on " << total << " faces on the Fermi surface of band " << this->iBand <<" =========\n\n\n" << std::endl;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::obtainGroupVelocitiesParallel(Bulk& bulk, int num_threads)  {
    std::cout << "Obtaining group velocities..." << std::endl;
    if (this->faceEigenstates.empty()) {
        std::cout << "ATTENTION: Velocity cannot be calculated." << std::endl;
        std::cout << "The face eigenstates and spins are empty, either call \"obtainFaceEigenstatesAndSpinsParallel\" first, or the Fermi surface is empty." << std::endl;
        return;
    }
    this->groupVelocities123.clear();
    this->groupVelocities123.resize(num_faces(this->theSurface));
    unsigned long long total = num_faces(this->theSurface);
    // First record all the coordinates of centroids.
    std::vector<arma::vec3> listOfCentroids;
    listOfCentroids.resize(this->groupVelocities123.size());
    std::cout << "======== Starting to fill velocities on "<< total << " faces on the Fermi surface of band " << this->iBand <<" =========" << std::endl;
    for(auto fi = this->theSurface.faces_begin(); fi != this->theSurface.faces_end(); ++fi) {
        Polyhedron::Halfedge_around_facet_circulator hfc = fi->facet_begin();
        K_exact::Point_3 p1 = hfc->vertex()->point();
        ++hfc;
        K_exact::Point_3 p2 = hfc->vertex()->point();
        ++hfc;
        K_exact::Point_3 p3 = hfc->vertex()->point();
        K_exact::Point_3 centroid = CGAL::centroid(p1, p2, p3);
        //std::cout << "debug three points obtained.\n";
        listOfCentroids[fi->id()] = arma::vec3{CGAL::to_double(centroid[0]), CGAL::to_double(centroid[1]), CGAL::to_double(centroid[2])};
    }
    std::cout << num_faces(this->theSurface) << " centroids recorded to be parallely processed." << std::endl;
    // Now, we obtain the group velocities using openMP:

    // First, create an atomic object to track the progress.
    unsigned long long completedIterations = 0;
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp single
        std::cout << "Number of OpenMP threads : " << omp_get_num_threads() << std::endl;
        unsigned long long localCounter = 0; // thread-local counter
        const auto reportInterval = std::max(1ULL, total / omp_get_num_threads() / 5);
#pragma omp for nowait
        for (unsigned long long i = 0; i < total; i++) {

            arma::vec3 k123 = listOfCentroids[i];
            std::vector<arma::cx_mat> DHDk = bulk.generateLocalDHDk(k123(0), k123(1), k123(2));
            arma::cx_vec psi = this->faceEigenstates[i];
            arma::cx_mat v0 = ((psi.t())*DHDk[0]*psi);
            arma::cx_mat v1 = ((psi.t())*DHDk[1]*psi);
            arma::cx_mat v2 = ((psi.t())*DHDk[2]*psi);
            arma::vec3 velocity = arma::vec3{real(v0(0,0)), real(v1(0,0)), real(v2(0,0))};
            this->groupVelocities123[i] = velocity;
            localCounter++;
            if (localCounter % reportInterval == 0) {
#pragma omp atomic
                completedIterations += localCounter;
                localCounter = 0; //Reset local counter after updating global counter.
                if (omp_get_thread_num() == 0) {
#pragma omp critical
                    std::cout << completedIterations << "/" << total << " velocities evaluated..." << std::endl;
                }
            }
        }
        if (localCounter > 0) {
#pragma omp atomic
            completedIterations += localCounter;
        }
    }
    std::cout << completedIterations << "/" << total << " velocities evaluated... all done." << std::endl;
    std::cout << "======== Velocities filled on " << total << " faces on the Fermi surface of band " << this->iBand <<" =========\n\n\n" << std::endl;

};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::calculateFaceSpins()   {
    if (this->faceEigenstates.empty())
        std::cout << "Err: faceEigenstates hasn't been calculated yet" << std::endl;
    else  {
        this->spins.clear();
        arma::uword matrixN = this->faceEigenstates[0].n_rows;
        arma::cx_mat I (matrixN/2, matrixN/2, arma::fill::eye);
        arma::cx_mat sigmaZ = arma::cx_mat{{std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)},
                                           {std::complex<double>(0.0, 0.0), std::complex<double>(-1.0, 0.0)}};

        arma::cx_mat sigmaX = arma::cx_mat{{std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0)},
                                           {std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)}};

        arma::cx_mat sigmaY = arma::cx_mat{{std::complex<double>(0.0, 0.0), std::complex<double>(0.0, -1.0)},
                                           {std::complex<double>(0.0, 1.0), std::complex<double>(0.0, 0.0)}};

        arma::cx_mat SigmaX, SigmaY, SigmaZ;
        SigmaX = kron(sigmaX, I);
        SigmaY = kron(sigmaY, I);
        SigmaZ = kron(sigmaZ, I);
        this->spins.clear();
        for (unsigned long long i=0; i<this->faceEigenstates.size(); i++)   {
            auto psi = this->faceEigenstates[i];
            double spinX = real(arma::as_scalar(psi.t()*SigmaX*psi));
            double spinY = real(arma::as_scalar(psi.t()*SigmaY*psi));
            double spinZ = real(arma::as_scalar(psi.t()*SigmaZ*psi));
            arma::vec3 spin{spinX, spinY, spinZ};
            this->spins.push_back(spin);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::writeSurfaceVtp(const std::string& filename, bool writeVelocity, bool writeSpin, bool writeBerryCurvature, bool binary) {
    std::cout << "Writing the surface mesh to " << filename << " for band " << this->iBand << "..." << std::flush;
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
    std::unordered_map<int, int> vertexIdToVtkIndex; // Map CGAL vertex ID to VTK index

    // Transfer vertices to VTK points and build the map
    int vtkIndex = 0;
    for (auto vit = this->theSurface.vertices_begin(); vit != this->theSurface.vertices_end(); ++vit, ++vtkIndex) {
        auto p = vit->point();
        arma::vec3 p123{CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
        arma::vec3 pxyz = this->B * p123;
        points->InsertNextPoint(pxyz(0), pxyz(1), pxyz(2));
        vertexIdToVtkIndex[vit->id()] = vtkIndex; // Map CGAL vertex ID to VTK index
    }

    // Transfer faces to VTK triangles using the mapping
    for (auto fit = this->theSurface.facets_begin(); fit != this->theSurface.facets_end(); ++fit) {
        vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
        Polyhedron::Halfedge_around_facet_circulator h = fit->facet_begin();
        for(int i = 0; i < 3; ++i, ++h) { // Since all faces are triangles
            int vtkPointIndex = vertexIdToVtkIndex[h->vertex()->id()];
            triangle->GetPointIds()->SetId(i, vtkPointIndex);
        }
        triangles->InsertNextCell(triangle);
    }

    polyData->SetPoints(points);
    polyData->SetPolys(triangles);

    // Optionally add velocity, spin, and berry curvature as face data
    if (writeVelocity) {
        vtkSmartPointer<vtkDoubleArray> velocities = vtkSmartPointer<vtkDoubleArray>::New();
        velocities->SetNumberOfComponents(3); // 3 components per vector
        velocities->SetName("Velocity");
        arma::mat33 transInvB = inv(this->B).t();
        for (const auto& velocity123 : groupVelocities123) {
            arma::vec3 velocityXyz = transInvB * velocity123;
            velocities->InsertNextTuple(velocityXyz.memptr());
        }
        polyData->GetCellData()->AddArray(velocities);
    }

    if (writeSpin) {
        vtkSmartPointer<vtkDoubleArray> spinsArray = vtkSmartPointer<vtkDoubleArray>::New();
        spinsArray->SetNumberOfComponents(3); // 3 components per vector
        spinsArray->SetName("Spin");
        for (const auto& spin : spins) {
            spinsArray->InsertNextTuple(spin.memptr());
        }
        polyData->GetCellData()->AddArray(spinsArray);
    }

    if (writeBerryCurvature) {
        vtkSmartPointer<vtkDoubleArray> berryCurvatures = vtkSmartPointer<vtkDoubleArray>::New();
        berryCurvatures->SetNumberOfComponents(1); // 1 component per scalar value
        berryCurvatures->SetName("BerryCurvature");
        for (const auto& curvature : berryCurvature) {
            berryCurvatures->InsertNextValue(curvature);
        }
        polyData->GetCellData()->AddArray(berryCurvatures);
    }

    // Write to VTK file
    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polyData);
    if (binary) {
        writer->SetDataModeToBinary();
    } else {
        writer->SetDataModeToAscii();
    }
    writer->Write();
    std::cout << "done." << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double FermiSurface::calculateEuclideanAreaOneTriangle(Polyhedron::Halfedge_around_facet_circulator hc) {
    K_exact::Point_3 p1 = hc->vertex()->point(); ++hc;
    K_exact::Point_3 p2 = hc->vertex()->point(); ++hc;
    K_exact::Point_3 p3 = hc->vertex()->point();
    K_exact::Vector_3 u = p2 - p1;
    K_exact::Vector_3 v = p3 - p1;
    arma::vec3 uxyz = CGAL::to_double(u[0])*this->b1 + CGAL::to_double(u[1])*this->b2 + CGAL::to_double(u[2])*this->b3;
    arma::vec3 vxyz = CGAL::to_double(v[0])*this->b1 + CGAL::to_double(v[1])*this->b2 + CGAL::to_double(v[2])*this->b3;
    return arma::norm(arma::cross(uxyz, vxyz))/2.0;
    //K_exact::Vector_3 cross_prod = CGAL::cross_product(u, v);
    // Calculate area of the transformed triangle
    //return std::sqrt(CGAL::to_double(cross_prod.squared_length())) / 2.0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::calculateEuclideanAreaForAllTriangles()  {
    this->triangleAreasXYZ.resize(num_faces(this->theSurface));
    for ( auto it = this->theSurface.facets_begin(); it != this->theSurface.facets_end(); ++it )   {
        Polyhedron::Halfedge_around_facet_circulator hc = it->facet_begin();
        double area = this->calculateEuclideanAreaOneTriangle(hc);
        triangleAreasXYZ[it->id()] = area;
    }
    std::cout << "Area calculated for " << num_faces(this->theSurface) << " triangles" << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::mat33 FermiSurface::velocityTensor() {
    // Here we need to go through all the surface pieces of the Fermi surface and calculate the integral based on Kubo formula.
    // First, check if velocities are already filled.
    if (this->groupVelocities123.empty() || this->triangleAreasXYZ.empty())  {
        std::cout << "Err: need to calculate velocities first. call the method "<< "\"" << "obtainGroupVelocitiesParallel" << "\"\n" << std::endl;
        arma::mat33 nothing(arma::fill::zeros);
        return nothing;
    }

    arma::mat33 velocityTensorIntegral(arma::fill::zeros);

    for (unsigned long long i = 0; i<this->groupVelocities123.size(); i++)    {
        arma::vec3 v123 = groupVelocities123[i];
        arma::mat33 transInvB = inv(this->B).t();
        arma::vec3 vxyz = transInvB * v123;
        arma::mat33 integrand = vxyz * vxyz.t() / arma::norm(vxyz) * this->triangleAreasXYZ[i];
        velocityTensorIntegral = velocityTensorIntegral + integrand;
    }
    //velocityTensorIntegral.print("debug tensor = ");
    return velocityTensorIntegral;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::mat33 FermiSurface::spinVelocityTensor() {
    arma::mat33 nothing(arma::fill::zeros);
    return nothing;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::processBorders() {
    unsigned long long halfedge_count = 0;
    for (auto eit = theSurface.halfedges_begin(); eit != theSurface.halfedges_end(); ++eit)  {
        if (eit->is_border()) {
            halfedge_count++;
        }
    }
    unsigned long long edge_count = 0;
    for (auto eit = theSurface.edges_begin(); eit != theSurface.edges_end(); ++eit)  {
        if (eit->is_border_edge()) {
            edge_count++;
        }
    }
    std::cout << "\ntotally we have " << num_halfedges(theSurface) << " halfedges and " << num_edges(theSurface) << " edges." << std::endl;
    std::cout << edge_count << " edges and " << halfedge_count << " half edges are found." << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::writeBoundaryEdgesVtu(const std::string& filename, bool binary) {
    std::cout << "Writing the boundary edges to " << filename << " for band " << this->iBand << "..." << std::flush;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

    std::unordered_map<int, int> vertexIdToVtkIndex; // Map CGAL vertex ID to VTK index

    // Transfer vertices to VTK points and build the map
    int vtkIndex = 0;
    for (auto vit = this->theSurface.vertices_begin(); vit != this->theSurface.vertices_end(); ++vit, ++vtkIndex) {
        auto p = vit->point();
        arma::vec3 p123{CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
        arma::vec3 pxyz = this->B * p123; // Assuming 'B' is a transformation matrix
        points->InsertNextPoint(pxyz(0), pxyz(1), pxyz(2));
        vertexIdToVtkIndex[vit->id()] = vtkIndex; // Map CGAL vertex ID to VTK index
    }

    // Transfer boundary edges to VTK lines using the mapping
    for (auto eit = this->theSurface.edges_begin(); eit != this->theSurface.edges_end(); ++eit) {
        if (eit->is_border_edge()) {
            vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
            int vtkPointIndexStart = vertexIdToVtkIndex[eit->prev()->vertex()->id()];
            int vtkPointIndexEnd = vertexIdToVtkIndex[eit->vertex()->id()];
            line->GetPointIds()->SetId(0, vtkPointIndexStart);
            line->GetPointIds()->SetId(1, vtkPointIndexEnd);
            lines->InsertNextCell(line);
        }
    }

    unstructuredGrid->SetPoints(points);
    unstructuredGrid->SetCells(VTK_LINE, lines);

    // Write to VTU file
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(unstructuredGrid);
    if (binary) {
        writer->SetDataModeToBinary();
    } else {
        writer->SetDataModeToAscii();
    }
    writer->Write();

    std::cout << "done." << std::endl;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FermiSurface::obtainBerryCurvatures(bool sharedBoundary) {
    if (this->triangleAreasXYZ.empty()) {
        return;
    }
    if (this->listOfVertexPhysics.empty()) {
        throw std::logic_error("Call obtainVertexEigenstatesParallel before calculating Berry curvature.");
    }
    if (sharedBoundary && (this->iBand != this->header.band_min || !this->hasSharedBoundary)) {
        throw std::invalid_argument("Shared-boundary integration requires the lowest band of a "
                                    "coincident multiplet with the same occupied side on this mesh.");
    }

    const auto faceCount = this->theSurface.size_of_facets();
    const auto width = this->header.M();
    const auto key = this->iBand - this->header.band_min;
    TrianglePhases phases;
    phases.bands = arma::regspace<arma::uvec>(this->header.band_min, this->header.band_max);
    phases.total.set_size(faceCount);
    phases.shares.set_size(faceCount, width);
    phases.minSingularValue.set_size(faceCount);
    phases.nearBranchCut.set_size(faceCount);
    phases.hallPhase.set_size(faceCount);
    phases.hallUsesConvention.zeros(faceCount);
    std::vector<double> curvature(faceCount);
    std::vector<double> flux(this->theSurface.size_of_vertices(), 0.0);
    std::size_t branchCutCount = 0;
    std::size_t conventionCount = 0;

    for (auto face = this->theSurface.facets_begin(); face != this->theSurface.facets_end(); ++face) {
        const auto id = face->id();
        std::array<std::size_t, 3> vertices;
        std::array<const arma::cx_mat*, 3> frames;
        std::array<const arma::vec*, 3> energies;
        bool keyDegenerate = false;
        auto edge = face->facet_begin();
        for (std::size_t i = 0; i < vertices.size(); ++i, ++edge) {
            vertices[i] = edge->vertex()->id();
            const auto representative = this->periodicConnectivity.vertexRepresentative[vertices[i]];
            const auto& physics = this->listOfVertexPhysics[representative];
            frames[i] = &physics.eigenvectors;
            energies[i] = &physics.energies;
            const auto& values = physics.energies;
            keyDegenerate |= (key > 0 && values(key) - values(key - 1) <= this->degeneracyTolerance) ||
                             (key + 1 < width && values(key + 1) - values(key) <= this->degeneracyTolerance);
        }

        try {
            const auto loop = quantumgeom::triangleWilsonPhases(frames, energies, this->degeneracyTolerance);
            phases.total(id) = loop.total;
            phases.shares.row(id) = loop.shares.t();
            phases.minSingularValue(id) = loop.minSingularValue;

            double hallPhase = loop.total;
            bool nearBranchCut = loop.nearBranchCut;
            if (!sharedBoundary && width > 1) {
                if (keyDegenerate) {
                    // The band label is ambiguous on this triangle. Use its share of
                    // the common loop as a convention and flag it for convergence checks.
                    hallPhase = loop.shares(key);
                    phases.hallUsesConvention(id) = 1;
                    ++conventionCount;
                } else {
                    // A resolved sheet needs its own rank-one projector. Transport
                    // of the whole group can cancel the sheet's internal Berry flux.
                    std::array<arma::cx_mat, 3> sheetFrames;
                    std::array<arma::vec, 3> sheetEnergies;
                    for (std::size_t i = 0; i < vertices.size(); ++i) {
                        sheetFrames[i] = frames[i]->col(key);
                        sheetEnergies[i] = {(*energies[i])(key)};
                        frames[i] = &sheetFrames[i];
                        energies[i] = &sheetEnergies[i];
                    }
                    const auto sheetLoop = quantumgeom::triangleWilsonPhases(frames, energies, 0.0);
                    hallPhase = sheetLoop.total;
                    nearBranchCut |= sheetLoop.nearBranchCut;
                }
            }
            phases.hallPhase(id) = hallPhase;
            phases.nearBranchCut(id) = nearBranchCut;
            branchCutCount += nearBranchCut;
            for (const auto vertex : vertices) {
                flux[vertex] += hallPhase;
            }
            curvature[id] = hallPhase / this->triangleAreasXYZ[id];
        } catch (const std::runtime_error& error) {
            throw std::runtime_error("Fermi-surface triangle " + std::to_string(id) + ": " + error.what());
        }
    }

    this->trianglePhases = std::move(phases);
    this->berryCurvature = std::move(curvature);
    this->berryFlux = std::move(flux);
    this->hallUsesSharedBoundary = sharedBoundary;
    if (branchCutCount) {
        std::cerr << "Warning: " << branchCutCount
                  << " Fermi-surface triangles have Berry phases near or across the principal branch cut. "
                     "Refine the mesh and check convergence.\n";
    }
    if (conventionCount) {
        std::cerr << "Warning: " << conventionCount
                  << " Fermi-surface triangles report a multiplet phase share for an unresolved band. "
                     "The Hall integral requires a resolved band or a shared multiplet.\n";
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::vec3 FermiSurface::haldaneBoundaryPhases() const {
    if (this->periodicConnectivity.seamPairs.empty()) {
        return arma::vec3(arma::fill::zeros);
    }
    const auto key = this->iBand - this->header.band_min;
    const auto& representatives = this->periodicConnectivity.vertexRepresentative;
    std::array<arma::cx_double, 3> products{1.0, 1.0, 1.0};
    for (const auto& seam : this->periodicConnectivity.seamPairs) {
        const auto& from = this->listOfVertexPhysics[representatives[seam.source]].eigenvectors;
        const auto& to = this->listOfVertexPhysics[representatives[seam.target]].eigenvectors;
        const auto link = this->hallUsesSharedBoundary
            ? quantumgeom::determinantBerryLink(from, to)
            : quantumgeom::normalizedBerryLink(arma::cdot(from.col(key), to.col(key)));

        // Each jump component weights a closed edge chain on the periodic mesh.
        // With the face-induced edge direction, -G * (-arg(link)) is the correction.
        for (std::size_t axis = 0; axis < products.size(); ++axis) {
            if (seam.latticeShift[axis] != 0) {
                products[axis] *= std::polar(1.0, std::arg(link) * static_cast<double>(seam.latticeShift[axis]));
            }
        }
    }
    return {std::arg(products[0]), std::arg(products[1]), std::arg(products[2])};
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::mat33 FermiSurface::intrinsicMatrixOfHaldaneIntegral() const {
    if (this->periodicConnectivity.unmatchedBorderHalfedges != 0) {
        throw std::runtime_error("Cannot evaluate the Haldane integral on a Fermi surface with unmatched periodic border edges.");
    }

    if (this->theSurface.size_of_facets() == 0) {
        return arma::mat33(arma::fill::zeros);
    }
    if (this->berryFlux.empty()) {
        throw std::logic_error("Call obtainBerryCurvature (Python) or obtainBerryCurvatures (C++) "
                               "before evaluating the Hall integral.");
    }
    if (arma::any(this->trianglePhases.hallUsesConvention != 0)) {
        throw std::runtime_error("The Hall integral requires Berry links for a resolved band or a shared "
                                 "multiplet. Refine the mesh near the degeneracy or use shared_boundary "
                                 "for a coincident multiplet.");
    }

    arma::vec3 integral(arma::fill::zeros);
    for (auto vertex = theSurface.vertices_begin(); vertex != theSurface.vertices_end(); ++vertex) {
        // The seam phases account for the jumps between the stored vertex lifts.
        const auto& p = vertex->point();
        const arma::vec3 k123{CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
        integral += this->berryFlux[vertex->id()] * k123;
    }
    integral /= 3.0; // Each triangle added its phase to all three vertices.
    integral += this->haldaneBoundaryPhases();
    // Surface orientation is defined in reduced coordinates; a reflected lattice reverses it.
    const double handedness = std::copysign(1.0, arma::det(this->B));
    const arma::vec3 cartesian = handedness * this->B * integral;
    const double ix = cartesian(0), iy = cartesian(1), iz = cartesian(2);
    return {{0.0, -iz, iy}, {iz, 0.0, -ix}, {-iy, ix, 0.0}};
}


FermiSurface::SheetSubspaceHeader FermiSurface::establishSubspace(
    Bulk& bulk, const std::vector<arma::vec3>& points, double gapTolerance, int numThreads) const {
    const auto bandCount = bulk.get_matrixN();
    if (bandCount == 1 || points.empty()) {
        return {this->iBand, this->iBand};
    }

    // Expand across any adjacent gap that closes anywhere on the sampled sheet.
    // Taking the minimum first also handles transitive crossings at different k points.
    const auto& representatives = this->periodicConnectivity.vertexRepresentative;
    const double infinity = std::numeric_limits<double>::infinity();
    arma::vec minimumGaps(bandCount - 1, arma::fill::value(infinity));
    int failed = 0;
    #pragma omp parallel num_threads(numThreads) reduction(|:failed)
    {
        arma::vec localGaps(bandCount - 1, arma::fill::value(infinity));
        #pragma omp for
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (representatives[i] != i) {
                continue;
            }
            const auto& k = points[i];
            const auto hamiltonian = bulk.generateLocalHamiltonian(k(0), k(1), k(2));
            arma::vec energies;
            if (!hamiltonian.is_finite() || !arma::eig_sym(energies, hamiltonian)) {
                failed = 1;
                continue;
            }
            localGaps = arma::min(localGaps, arma::diff(energies));
        }
        #pragma omp critical
        minimumGaps = arma::min(minimumGaps, localGaps);
    }
    if (failed) {
        throw std::runtime_error("Could not determine the fixed Fermi-surface band range.");
    }

    SheetSubspaceHeader bands{this->iBand, this->iBand};
    while (bands.band_min > 0 && minimumGaps(bands.band_min - 1) <= gapTolerance) {
        --bands.band_min;
    }
    while (bands.band_max + 1 < bandCount && minimumGaps(bands.band_max) <= gapTolerance) {
        ++bands.band_max;
    }
    return bands;
}

bool FermiSurface::checkSharedBoundary(Bulk& bulk, const std::vector<FermiVertexPhysics>& physics,
                                       SheetSubspaceHeader bands, double tolerance) const {
    // First reject distinct sheets using the energies already available at the vertices.
    for (const auto& vertex : physics) {
        if (!vertex.energies.empty() && vertex.energies.back() - vertex.energies.front() > tolerance) {
            return false;
        }
    }

    for (auto face = this->theSurface.facets_begin(); face != this->theSurface.facets_end(); ++face) {
        std::array<arma::vec3, 3> points;
        auto edge = face->facet_begin();
        for (auto& point : points) {
            const auto& p = edge->vertex()->point();
            point = {CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
            ++edge;
        }
        const arma::vec3 normal = arma::cross(points[1] - points[0], points[2] - points[0]);
        const double length = arma::norm(normal);
        if (length == 0.0) {
            return false;
        }

        const arma::vec3 k = (points[0] + points[1] + points[2]) / 3.0;
        const auto hamiltonian = bulk.generateLocalHamiltonian(k(0), k(1), k(2));
        arma::vec energies;
        arma::cx_mat eigenvectors;
        if (!hamiltonian.is_finite() || !arma::eig_sym(energies, eigenvectors, hamiltonian)) {
            throw std::runtime_error("Could not diagonalize a Fermi-surface face centroid Hamiltonian.");
        }
        if (energies(bands.band_max) - energies(bands.band_min) > tolerance ||
            (bands.band_min > 0 && energies(bands.band_min) - energies(bands.band_min - 1) <= tolerance) ||
            (bands.band_max + 1 < energies.n_elem &&
             energies(bands.band_max + 1) - energies(bands.band_max) <= tolerance)) {
            return false;
        }

        // At a degeneracy the projected normal derivative gives the slopes of
        // the individual branches. Their signs must put the occupied states
        // on the same side of the shared surface.
        const auto derivatives = bulk.generateLocalDHDk(k(0), k(1), k(2));
        const arma::cx_mat derivative =
            (normal(0) * derivatives[0] + normal(1) * derivatives[1] + normal(2) * derivatives[2]) / length;
        const arma::cx_mat frame = eigenvectors.cols(bands.band_min, bands.band_max);
        const arma::cx_mat projected = frame.t() * derivative * frame;
        arma::vec slopes;
        if (!projected.is_finite() || !arma::eig_sym(slopes, (projected + projected.t()) / 2.0)) {
            throw std::runtime_error("Could not resolve the normal slopes of a candidate shared Fermi surface.");
        }
        const double slopeTolerance =
            64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, arma::norm(projected, "fro"));
        if (slopes.front() <= slopeTolerance && slopes.back() >= -slopeTolerance) {
            return false;
        }
    }
    return true;
}
