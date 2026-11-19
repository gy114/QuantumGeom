/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "FermiSurfaceExtraction.hpp"
#include "WriteToVTK.hpp"

#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <iostream>
#include <map>
#include <unordered_set>
#include <armadillo>
#include <omp.h>
#include <set>
#include <stdexcept>
#include <utility>


#include "WriteToVTK.hpp"
#include "Bulk.hpp"
//class Bulk;

namespace {

std::vector<arma::vec> evaluateMeshEnergies(Bulk& bulk, const std::vector<P3_Point>& points, int numThreads) {
    if (numThreads < 1) {
        throw std::invalid_argument("Mesh energy evaluation requires a positive thread count.");
    }
    std::vector<arma::vec> energies(points.size());
    int failed = 0;
    #pragma omp parallel for num_threads(numThreads) reduction(|:failed)
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& k = points[i];
        const auto hamiltonian = bulk.generateLocalHamiltonian(k.x(), k.y(), k.z());
        if (!hamiltonian.is_finite() || !arma::eig_sym(energies[i], hamiltonian)) {
            failed = 1;
        }
    }
    if (failed) {
        throw std::runtime_error("Could not calculate the eigenvalues of a bulk mesh vertex.");
    }
    return energies;
}

void setMeshEnergies(VertexData& vertex, arma::vec&& energies) {
    vertex.eigVal = std::move(energies);
    // Insertion can find an existing vertex whose exact energies were already cached.
    vertex.isExact = false;
    vertex.exactEnergy.clear();
}

std::size_t berryRefinementCount(std::size_t total, int refineDepth) {
    const auto percent = static_cast<std::size_t>(std::clamp(refineDepth, 0, 100));
    return (total / 100) * percent + ((total % 100) * percent) / 100;
}

BulkBandEigenstates evaluateBerryRefinementStates(
    P3_Delaunay& mesh, Bulk& bulk, const std::vector<P3_Delaunay::Cell_handle>& cells,
    arma::uword band, int numThreads) {
    BulkBandEigenstates states;
    // Retain one column only, and evaluate each periodic vertex touched by the candidates once.
    std::vector<std::pair<P3_Delaunay::Vertex_handle, arma::cx_vec*>> pending;
    for (const auto& cell : cells) {
        for (int i = 0; i < 4; ++i) {
            const auto vertex = mesh.get_original_vertex(cell->vertex(i));
            const auto [entry, added] = states.try_emplace(vertex);
            if (added) {
                pending.emplace_back(vertex, &entry->second);
            }
        }
    }

    int failed = 0;
    #pragma omp parallel for num_threads(numThreads) reduction(|:failed)
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const auto& [vertex, state] = pending[i];
        const auto& k = vertex->point();
        const auto hamiltonian = bulk.generateLocalHamiltonian(k.x(), k.y(), k.z());
        arma::vec energies;
        arma::cx_mat eigenvectors;
        if (!hamiltonian.is_finite() || !arma::eig_sym(energies, eigenvectors, hamiltonian)) {
            failed = 1;
        } else {
            *state = eigenvectors.col(band);
        }
    }
    if (failed) {
        throw std::runtime_error("Could not calculate eigenstates for Berry-guided mesh refinement.");
    }
    return states;
}

} // namespace


typedef std::array<K_exact::Point_3, 3> Triangle;
//typedef std::array<P3_Point, 3> Triangle;

bool isCorrectlyOriented(const Triangle& tri, P3_Delaunay::Cell_handle cell, const K_exact::FT value, K_exact::Point_3 referenceVertex) {
//bool isCorrectlyOriented(const Triangle& tri, P3_Delaunay::Cell_handle cell, Iso_cuboid domain, const double value, P3_Point referenceVertex) {
    // Calculate the normal of the adjusted triangle using the right-hand rule
    K_exact::Vector_3 u = tri[1] - tri[0];
    K_exact::Vector_3 v = tri[2] - tri[0];
    K_exact::Vector_3 normal = CGAL::cross_product(u, v);

    // Vector from the first adjusted triangle vertex to the adjusted reference vertex
    K_exact::Vector_3 toReference = referenceVertex - tri[0];

    return (value <= 0) ? (normal * toReference) > 0 : (normal * toReference) < 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<Triangle> extractIntersectingTriangles(P3_Delaunay::Cell_handle cell, P3_Delaunay& T, arma::uword iBand, K_exact::FT Ef) {
    std::vector<Triangle> intersectingTriangles;

    K_exact::FT dx = K_exact::FT(1);
    K_exact::FT dy = K_exact::FT(1);
    K_exact::FT dz = K_exact::FT(1);

    std::array<K_exact::FT, 4> values;
    for (int i = 0; i < 4; ++i)  {
        if (!cell->vertex(i)->info().isExact) {
            std::vector<K_exact::FT> newExactEnergy;
            for (arma::uword j=0; j<cell->vertex(i)->info().eigVal.n_rows; j++) {
                newExactEnergy.emplace_back(K_exact::FT(cell->vertex(i)->info().eigVal(j)));
            }
            K::Point_3 p = cell->vertex(i)->point();
            cell->vertex(i)->info().exactEnergy = newExactEnergy;
            cell->vertex(i)->info().exactPoint = K_exact::Point_3{K_exact::FT(p.x()), K_exact::FT(p.y()), K_exact::FT(p.z())};
            values[i] = newExactEnergy[iBand] - Ef;
            cell->vertex(i)->info().isExact = true;
        }
        else  {
            values[i] = cell->vertex(i)->info().exactEnergy[iBand] - Ef;
        }
    }
    // Check for sign changes
    bool isIntersecting = false;

    for (int i = 0; i < values.size() - 1 && !isIntersecting; ++i) {
        for (int j = i + 1; j < values.size(); ++j) {
            if (values[i] * values[j] < 0) {
                isIntersecting = true;
                break; // No need to continue checking once an intersection is found
            }
        }
    }

    if (!isIntersecting) {
        // No intersection or ambiguous case
        return intersectingTriangles;
    }

    // Determine the intersecting edges and compute intersection points
    std::vector<K_exact::Point_3> intersectionPoints;
    //std::vector<P3_Point> intersectionPoints;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (values[i] * values[j] < 0) { // Sign change implies intersection
                // Linear interpolation to find the intersection point
                K_exact::FT ratio = CGAL::abs(values[i]) / (CGAL::abs(values[i]) + CGAL::abs(values[j]));
                PeriodicPoint pp1 = T.periodic_point(cell, i);
                PeriodicPoint pp2 = T.periodic_point(cell, j);
                CGAL::Periodic_3_offset_3 offset1 = pp1.second;
                CGAL::Periodic_3_offset_3 offset2 = pp2.second;

                K_exact::Point_3 p1 = cell->vertex(i)->info().exactPoint;
                K_exact::Point_3 p2 = cell->vertex(j)->info().exactPoint;

                K_exact::FT x1 = p1.x() + offset1[0] * dx;
                K_exact::FT y1 = p1.y() + offset1[1] * dy;
                K_exact::FT z1 = p1.z() + offset1[2] * dz;

                K_exact::FT x2 = p2.x() + offset2[0] * dx;
                K_exact::FT y2 = p2.y() + offset2[1] * dy;
                K_exact::FT z2 = p2.z() + offset2[2] * dz;

                K_exact::Point_3 actual_point1(x1, y1, z1);
                K_exact::Point_3 actual_point2(x2, y2, z2);
                //P3_Point actual_point1(x1, y1, z1);
                //P3_Point actual_point2(x2, y2, z2);

                K_exact::Point_3 intersection = CGAL::barycenter(actual_point1, 1 - ratio, actual_point2, ratio);
                //P3_Point P3_intersection = intersection;
                //P3_Point P3_intersection = CGAL::barycenter(actual_point1, 1 - ratio, actual_point2, ratio);
                // now we output the offset and see if it is properly handled.

                intersectionPoints.push_back(intersection);
            }
        }
    }

    // Now form triangles from intersection points and orient them
    // First, we record the actual position of vertex(0) in the cell. We will later pass it to another function to orient the triangles.
    PeriodicPoint refpp = T.periodic_point(cell, 0);
    CGAL::Periodic_3_offset_3 offset = refpp.second;
    K_exact::Point_3 refppp = cell->vertex(0)->info().exactPoint;
    K_exact::FT refX = refppp.x() + offset[0] * dx;
    K_exact::FT refY = refppp.y() + offset[1] * dy;
    K_exact::FT refZ = refppp.z() + offset[2] * dz;
    K_exact::Point_3 referenceVertex{refX, refY, refZ};
    //P3_Point referenceVertex{refX, refY, refZ};
    if (intersectionPoints.size() == 3) { // One triangle
        Triangle tri = {intersectionPoints[0], intersectionPoints[1], intersectionPoints[2]};
        if (!isCorrectlyOriented(tri, cell, values[0], referenceVertex)) {
            std::swap(tri[1], tri[2]); // Swap two vertices to correct orientation
        }
        intersectingTriangles.push_back(tri);
        return intersectingTriangles;
    } else if (intersectionPoints.size() == 4) { // Two triangles
        // Form the first triangle with the first three points
        Triangle tri1 = {intersectionPoints[0], intersectionPoints[1], intersectionPoints[2]};

        // Form the second triangle with the last three points
        Triangle tri2 = {intersectionPoints[1], intersectionPoints[2], intersectionPoints[3]};
        // the order of checking 0-1, 0-2, 0-3, 1-2, 1-3, 2-3 makes sure that the first three and the last three must share a diagonal going through the cell, instead of on one of the four faces.
        if (!isCorrectlyOriented(tri1, cell, values[0], referenceVertex)) {
            std::swap(tri1[1], tri1[2]); // Swap two vertices to correct orientation
        }
        if (!isCorrectlyOriented(tri2, cell, values[0], referenceVertex)) {
            std::swap(tri2[1], tri2[2]); // Swap two vertices to correct orientation
        }

        intersectingTriangles.push_back(tri1);
        intersectingTriangles.push_back(tri2);
        return intersectingTriangles;
    }

    std::cout << "warning! invalid piece of iso surface found.\n";
    return intersectingTriangles;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void refineMeshParallel(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef, int num_OpenMP_threads)   {
    // This function goes through all the unique cell.
    // If one cell is found to be close enough to zero, then collect.
    // After that we go through all the collected unique-edge handles, and insert a P3_Point at the centroid, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique edges can be outside the cuboid.

    if (mesh.number_of_vertices() == 0) {// In case if the mesh was not initialized.
        std::cout << "Err: the mesh is not initialized. Need to call \"initializeMeshParallel\" before this." << std::endl;
        return;
    }
    if (iBand >= bulk.get_matrixN()) {
        throw std::invalid_argument("Bulk mesh refinement band index is out of range.");
    }

    if (!mesh.is_1_cover()) {
        mesh.convert_to_1_sheeted_covering();
    }
    // Start to refine.

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {

        // Now go through all unique edges.
        // If a unique edge is close enough to zero, collect its middle point.
        //std::vector<P3_Delaunay::Edge> edges_to_refine;

        std::vector<K::Point_3> points_to_insert;
        double maxErr = 0.0;
        double totalErr = 0.0;
        unsigned long numberOfCrossingEdges = 0;

        for (auto cit = mesh.unique_cells_begin(); cit != mesh.unique_cells_end(); ++cit) {
            double max_f = -std::numeric_limits<double>::infinity();
            double min_f = std::numeric_limits<double>::infinity();

            for (int i = 0; i < 4; ++i) { // Each cell has 4 vertices
                const auto vertex = mesh.get_original_vertex(cit->vertex(i));
                const double f = vertex->info().eigVal(iBand) - Ef;
                max_f = std::max(max_f, f);
                min_f = std::min(min_f, f);
            }
            // Calculate t using max_f and min_f
            double t;
            if (max_f != min_f) { // Ensure denominator is not zero
                t = (0 - min_f) / (max_f - min_f);
            } else {
                // If max_f == min_f, we cannot determine a zero-crossing in the traditional sense
                continue; // Move to the next cell
            }


            // Check if the zero-crossing is within the edge or within the extended range
            if (-2.0 < t && t < 3.0) {
                // Calculate the "real" positions of the vertices considering their offsets
                double sum_x = 0, sum_y = 0, sum_z = 0;
                int count = 0;

                for (int i = 0; i < 4; ++i) { // Each cell has 4 vertices
                    PeriodicPoint pp = mesh.periodic_point(cit, i);
                    P3_Point real_point = mesh.point(pp); // Converts periodic point to its real position
                    sum_x += real_point.x();
                    sum_y += real_point.y();
                    sum_z += real_point.z();
                    count++;
                }

                // Calculate the centroid coordinates
                double centroid_x = sum_x / count;
                double centroid_y = sum_y / count;
                double centroid_z = sum_z / count;
                P3_Point centroid(centroid_x, centroid_y, centroid_z);
                P3_Traits traits(mesh.domain());

                // New internal name in recent CGAL
                P3_Point canonical = CGAL::P3T3::internal::robust_canonicalize_point(centroid, traits);

                // Map to the periodic triangulation's canonical representation
                P3_Point new_point = mesh.construct_point(canonical);

                points_to_insert.push_back(new_point);
                // now we check the maxErr and diviation.
                if (t > 0.0 && t < 1.0)   {
                    totalErr += std::abs(max_f-min_f);
                    numberOfCrossingEdges++;
                    if (std::abs(max_f-min_f) > maxErr)
                        maxErr = std::abs(max_f-min_f);
                }
            }
        }
        std::cout << "Calculating eigenvalues for " << points_to_insert.size()
                  << " new points." << std::endl;
        auto energies = evaluateMeshEnergies(bulk, points_to_insert, num_OpenMP_threads);
        std::cout << "Before inserting, the mesh has " << mesh.number_of_vertices()
                  << " vertices..." << std::endl;
        for (std::size_t i = 0; i < points_to_insert.size(); ++i) {
            try {
                const auto vertex = mesh.get_original_vertex(mesh.insert(points_to_insert[i]));
                setMeshEnergies(vertex->info(), std::move(energies[i]));
            } catch (const std::exception& error) {
                std::cerr << "An error occurred during insertion: " << error.what() << std::endl;
            }
        }
        std::cout << "Refinement is done for Round " << iTimes+1 << ", standard div=" << totalErr/(double)numberOfCrossingEdges << ", maxErr=" << maxErr << std::endl;
    }
    //write_to_vtu_P3("refinedMesh.vtu", mesh, domain, iBand, true, bulk.get_B());
    //std::cout << "Refined mesh written in refinedMesh.vtu\n";
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void berryCurvatureSingleCell(P3_Delaunay& mesh, P3_Delaunay::Cell_handle& ch,
                              const BulkBandEigenstates& states, arma::vec4& outputBerryCurvature, arma::mat33& BB) {
    for (int i=0; i<4; i++)   {
        // Note that vertex_triple_index returns values following CGAL's ccw convention. We need to outwards pointing flux, which is opposite to it.
        // That is why we do 012->210 in the following.
        int ind0 = CGAL::Triangulation_utils_3::vertex_triple_index(i, 2);
        int ind1 = CGAL::Triangulation_utils_3::vertex_triple_index(i, 1);
        int ind2 = CGAL::Triangulation_utils_3::vertex_triple_index(i, 0);
        const auto& psi0 = states.at(mesh.get_original_vertex(ch->vertex(ind0)));
        const auto& psi1 = states.at(mesh.get_original_vertex(ch->vertex(ind1)));
        const auto& psi2 = states.at(mesh.get_original_vertex(ch->vertex(ind2)));
        const auto overlap = arma::cdot(psi0, psi1) * arma::cdot(psi1, psi2) * arma::cdot(psi2, psi0);

        PeriodicPoint pp0 = mesh.periodic_point(ch, ind0);
        PeriodicPoint pp1 = mesh.periodic_point(ch, ind1);
        PeriodicPoint pp2 = mesh.periodic_point(ch, ind2);
        K::Point_3 p0 = mesh.point(pp0);
        K::Point_3 p1 = mesh.point(pp1);
        K::Point_3 p2 = mesh.point(pp2);

        arma::vec3 ppp0 = arma::vec3{p0.x(), p0.y(), p0.z()};
        arma::vec3 ppp1 = arma::vec3{p1.x(), p1.y(), p1.z()};
        arma::vec3 ppp2 = arma::vec3{p2.x(), p2.y(), p2.z()};

        arma::vec3 u = ppp1 - ppp0;
        arma::vec3 v = ppp2 - ppp0;
        arma::vec3 cross_product = arma::cross(BB * u, BB * v);
        double area = arma::norm(cross_product) / 2.0;
        outputBerryCurvature(i) = -std::imag(overlap) / area;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<P3_Delaunay::Cell_handle> topListOfLargeBerryCurvatures(
    std::vector<P3_Delaunay::Cell_handle>& fullList, P3_Delaunay& mesh,
    const BulkBandEigenstates& states, arma::mat33& BB, int refineDepth) {
    const size_t total = fullList.size();
    const size_t numberToSelect = berryRefinementCount(total, refineDepth);

    if (numberToSelect == 0) {
        return {};
    }
    if (numberToSelect == total) {
        return fullList;
    }

    std::vector<std::pair<double, P3_Delaunay::Cell_handle>> listOfPairs;
    listOfPairs.reserve(total);
    for (auto ch : fullList) {
        arma::vec4 berryCurvature;
        berryCurvatureSingleCell(mesh, ch, states, berryCurvature, BB);
        std::pair<double, P3_Delaunay::Cell_handle> newPair(arma::max(arma::abs(berryCurvature)), ch);
        listOfPairs.push_back(newPair);
    }
    const size_t nThresholdIndex = total - numberToSelect;
    std::nth_element(listOfPairs.begin(), listOfPairs.begin() + nThresholdIndex, listOfPairs.end(),
                     [](const std::pair<double, P3_Delaunay::Cell_handle> &a, const std::pair<double, P3_Delaunay::Cell_handle>& b) {
                         return a.first < b.first;
                     });
    std::vector<P3_Delaunay::Cell_handle> result;
    result.reserve(numberToSelect);
    for (size_t i = nThresholdIndex; i < total; ++i)  {
        result.push_back(listOfPairs[i].second);
    }
    return result;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void refineMeshParallelForBerryPhaseImplementation(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef, int num_OpenMP_threads, int refineDepth)   {
    // This function goes through all the unique cells.
    // If one unique edge is found to be close enough to zero, and it may have large berry phase on one of its facets, then collect.
    // After that we go through all the collected cell handles, and insert a P3_Point at the centroid of the cell, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique edges can be outside the cuboid.

    if (mesh.number_of_vertices() == 0) {// In case if the mesh was not initialized.
        std::cout << "Err: the mesh is not initialized. Need to call \"initializeMeshParallel\" before this." << std::endl;
        return;
    }
    if (iBand >= bulk.get_matrixN() || num_OpenMP_threads < 1) {
        throw std::invalid_argument("Berry-guided refinement requires a valid band and positive thread count.");
    }

    if (!mesh.is_1_cover()) {
        mesh.convert_to_1_sheeted_covering();
    }
    // Start to refine.

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {

        // Now go through all unique edges.
        // If a unique edge is close enough to zero, collect its middle point.
        //std::vector<P3_Delaunay::Edge> edges_to_refine;

        std::vector<K::Point_3> points_to_insert;
        std::vector<P3_Delaunay::Cell_handle> listOfCells;

        for (auto cit = mesh.unique_cells_begin(); cit != mesh.unique_cells_end(); ++cit) {
            double max_f = -std::numeric_limits<double>::infinity();
            double min_f = std::numeric_limits<double>::infinity();

            for (int i = 0; i < 4; ++i) { // Each cell has 4 vertices
                const auto vertex = mesh.get_original_vertex(cit->vertex(i));
                const double f = vertex->info().eigVal(iBand) - Ef;
                max_f = std::max(max_f, f);
                min_f = std::min(min_f, f);
            }
            // Calculate t using max_f and min_f
            double t;
            if (max_f != min_f) { // Ensure denominator is not zero
                t = (0 - min_f) / (max_f - min_f);
            } else {
                // If max_f == min_f, we cannot determine a zero-crossing in the traditional sense
                continue; // Move to the next cell
            }


            // Check if the zero-crossing is within the edge or within the extended range
            if (-2.0 < t && t < 3.0) {
                listOfCells.push_back(cit);
            }
        }

        arma::mat33 BB = bulk.get_B();
        std::vector<P3_Delaunay::Cell_handle> topListOfCells;
        {
            BulkBandEigenstates states;
            const auto selectedCount = berryRefinementCount(listOfCells.size(), refineDepth);
            if (selectedCount > 0 && selectedCount < listOfCells.size()) {
                states = evaluateBerryRefinementStates(mesh, bulk, listOfCells, iBand, num_OpenMP_threads);
            }
            topListOfCells = topListOfLargeBerryCurvatures(listOfCells, mesh, states, BB, refineDepth);
        }
        // Now we get the centroids of these cells.
        for (auto ch : topListOfCells)  {

            double sum_x = 0, sum_y = 0, sum_z = 0;
            int count = 0;

            for (int i = 0; i < 4; ++i) { // Each cell has 4 vertices
                PeriodicPoint pp = mesh.periodic_point(ch, i);
                P3_Point real_point = mesh.point(pp); // Converts periodic point to its real position
                sum_x += real_point.x();
                sum_y += real_point.y();
                sum_z += real_point.z();
                count++;
            }

            // Calculate the centroid coordinates
            double centroid_x = sum_x / count;
            double centroid_y = sum_y / count;
            double centroid_z = sum_z / count;
            P3_Point centroid(centroid_x, centroid_y, centroid_z);
            P3_Traits traits(mesh.domain());
            P3_Point new_point = CGAL::P3T3::internal::robust_canonicalize_point(centroid, traits);
            new_point = mesh.construct_point(new_point);
            points_to_insert.push_back(new_point);
        }


        std::cout << "Calculating eigenvalues for " << points_to_insert.size()
                  << " new points." << std::endl;
        auto energies = evaluateMeshEnergies(bulk, points_to_insert, num_OpenMP_threads);
        std::cout << "Before inserting, the mesh has " << mesh.number_of_vertices()
                  << " vertices..." << std::endl;
        for (std::size_t i = 0; i < points_to_insert.size(); ++i) {
            try {
                const auto vertex = mesh.get_original_vertex(mesh.insert(points_to_insert[i]));
                setMeshEnergies(vertex->info(), std::move(energies[i]));
            } catch (const std::exception& error) {
                std::cerr << "An error occurred during insertion: " << error.what() << std::endl;
            }
        }
        std::cout << "Refinement is done for Round " << iTimes+1 << std::endl;
    }
    //write_to_vtu_P3("refinedMesh.vtu", mesh, domain, iBand, true, bulk.get_B());
    //std::cout << "Refined mesh written in refinedMesh.vtu\n";
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
P3_Delaunay initializeMeshParallel(Bulk& bulk, const unsigned long num_initial_points, int num_OpenMP_threads) {
    // Now generate a random, periodic delaunay mesh in the cuboid (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5).
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    double dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();
    arma::vec3 b1 = bulk.get_b1();
    arma::vec3 b2 = bulk.get_b2();
    arma::vec3 b3 = bulk.get_b3();
    arma::mat33 B = bulk.get_B(); // This is used to transfer k123 to kxyz coordinates.
    arma::mat33 invB = arma::inv(B); // This is to transfer kxyz to k123.
    // Now we find the smallest box that bounds the parallelepiped.
    std::vector<arma::vec3> vertices = {
            -0.5*b1 -0.5*b2 -0.5*b3,
            +0.5*b1 -0.5*b2 -0.5*b3,
            -0.5*b1 +0.5*b2 -0.5*b3,
            -0.5*b1 -0.5*b2 +0.5*b3,
            +0.5*b1 +0.5*b2 -0.5*b3,
            -0.5*b1 +0.5*b2 +0.5*b3,
            +0.5*b1 -0.5*b2 +0.5*b3,
            +0.5*b1 +0.5*b2 +0.5*b3,
    };
    arma::vec3 minVec = vertices[0];
    arma::vec3 maxVec = vertices[0];

// Find min and max for each axis
    for (const auto& vertex : vertices) {
        for (int i = 0; i < 3; ++i) {
            minVec(i) = std::min(minVec(i), vertex(i));
            maxVec(i) = std::max(maxVec(i), vertex(i));
        }
    }

// Now set up the distributions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disX(minVec(0), maxVec(0));
    std::uniform_real_distribution<> disY(minVec(1), maxVec(1));
    std::uniform_real_distribution<> disZ(minVec(2), maxVec(2));

    P3_Delaunay T(domain);  // Create an empty P3_Delaunay triangulation.
    // Now let's generate the points in the bounding box uniformly random.
    // If the points are inside the domain after transferred to 123 coordinate, then include.
    // Otherwise, keep generating.
    //std::fstream debugOutput("debugPoint.txt", std::ios::out);
    unsigned long generatedN = 0;
    std::vector<P3_Point> listOfPoints;
    std::vector<P3_Delaunay::Vertex_handle> listOfVertexHandles;
    while(generatedN < num_initial_points) {
        arma::vec3 randPointXYZ = {disX(gen), disY(gen), disZ(gen)};

        arma::vec3 randPoint123 = invB * randPointXYZ;
        //randPoint123.print("debug 123 = ");
        // Use domain's boundaries for checking
        if (randPoint123(0) > domain.xmin() && randPoint123(0) < domain.xmax() &&
            randPoint123(1) > domain.ymin() && randPoint123(1) < domain.ymax() &&
            randPoint123(2) > domain.zmin() && randPoint123(2) < domain.zmax()) {
            // Insert the point into the Delaunay triangulation
            P3_Delaunay::Vertex_handle vh = T.insert(P3_Point(randPoint123(0), randPoint123(1), randPoint123(2)));
            listOfPoints.push_back(vh->point());
            listOfVertexHandles.push_back(vh);
            generatedN++;
            //        debugOutput << randPointXYZ(0) << " " << randPointXYZ(1) << " " << randPointXYZ(2) << " " << 1.0 << "\n";
            //  } else {
            //      debugOutput << randPointXYZ(0) << " " << randPointXYZ(1) << " " << randPointXYZ(2) << " " << 0.0 << "\n";
        }
    }

    std::cout << "Initializing " << T.number_of_vertices() << " vertices with "
              << num_OpenMP_threads << " OpenMP threads." << std::endl;
    auto energies = evaluateMeshEnergies(bulk, listOfPoints, num_OpenMP_threads);
    for (std::size_t i = 0; i < listOfVertexHandles.size(); ++i) {
        const auto vertex = T.get_original_vertex(listOfVertexHandles[i]);
        setMeshEnergies(vertex->info(), std::move(energies[i]));
    }
    std::cout << energies.size() << "/" << energies.size() << " vertices evaluated. All done." << std::endl;
    std::cout << "Initialization of " << T.number_of_vertices() << " vertices finished.\n\n" << std::endl;
    //write_to_vtu_P3("meshAfterInitialization.vtu", T, domain, 0, true, B);
    return T;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Polyhedron extractOneFermiSurfaceMesh(Bulk& bulk, arma::uword iBand, K_exact::FT Ef) {

    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    std::vector<K_exact::Point_3> points; // Global list of points (may contain duplicates initially)
    std::vector<std::vector<std::size_t>> triangles; // Global list of triangles (using indices of points)
    std::cout << "generating surface mesh ..." << std::flush;
    for (auto cl = bulk.theMesh.unique_cells_begin(); cl != bulk.theMesh.unique_cells_end(); ++cl)
    {
        auto trianglesInThisCell = extractIntersectingTriangles(cl, bulk.theMesh, iBand, Ef);
        if (!trianglesInThisCell.empty()) {
            //std::cout << "debug " << trianglesInThisCell.size() << " triangles found.\n";
            for (const auto& tri : trianglesInThisCell) {
                // Insert points from the triangle into the global list
                std::vector<std::size_t> triIndices;
                for (const auto& p : tri) {
                    points.push_back(p); // Add point to global list
                    triIndices.push_back(points.size() - 1); // Use current index for the triangle
                }
                triangles.push_back(triIndices); // Add triangle (indices) to global list
            }
        }
    }

    std::cout << "done." << std::endl;
    std::cout << "Repairing the triangle soup... " << std::flush;
    //CGAL::Polygon_mesh_processing::repair_polygon_soup(points, triangles);
    CGAL::Polygon_mesh_processing::merge_duplicate_points_in_polygon_soup(points, triangles);
    std::cout << "done." << std::endl;
    // Then, since triangles are already oriented, we merge the soup to create the Polyhedron_3
    Polyhedron isoSurface;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, triangles, isoSurface);
    isoSurface.normalize_border();
    bool isValid = isoSurface.is_valid();
    if (isValid)
        std::cout << "The surface mesh is valid. " << std::endl;
    else
        std::cout << "Warning: The surface mesh is not valid." << std::endl;
    return isoSurface;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace {

struct WrappedVertex {
    K_exact::Point_3 point;
    std::array<long long, 3> latticeShift;
};

WrappedVertex wrapVertex(const K_exact::Point_3& p) {
    const K_exact::FT half = K_exact::FT(1) / K_exact::FT(2);
    std::array<long long, 3> shift{};

    auto wrapCoordinate = [half](K_exact::FT coordinate, long long& shift) {
        while (coordinate < -half) {
            coordinate += K_exact::FT(1);
            --shift;
        }
        while (coordinate >= half) {
            coordinate -= K_exact::FT(1);
            ++shift;
        }
        return coordinate;
    };

    const auto x = wrapCoordinate(p.x(), shift[0]);
    const auto y = wrapCoordinate(p.y(), shift[1]);
    const auto z = wrapCoordinate(p.z(), shift[2]);
    return {K_exact::Point_3(x, y, z), shift};
}

}  // namespace

K_exact::Point_3 map_to_exact_cuboid(const K_exact::Point_3& p) {
    return wrapVertex(p).point;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PeriodicSurfaceConnectivity buildPeriodicSurfaceConnectivity(const Polyhedron& surface) {
    using Point = K_exact::Point_3;
    PeriodicSurfaceConnectivity result;
    result.vertexRepresentative.resize(surface.size_of_vertices());
    std::vector<WrappedVertex> wrappedVertices(surface.size_of_vertices());
    std::map<Point, std::size_t> representatives;
    // A periodic vertex can have several stored images, all outside the zone.
    for (auto vertex = surface.vertices_begin(); vertex != surface.vertices_end(); ++vertex) {
        const auto id = vertex->id();
        wrappedVertices[id] = wrapVertex(vertex->point());
        result.vertexRepresentative[id] = representatives.emplace(wrappedVertices[id].point, id).first->second;
    }

    using DirectedEdge = std::pair<Point, Point>;
    using VertexPair = std::pair<std::size_t, std::size_t>;
    struct BorderCandidates {
        std::array<std::size_t, 2> counts{};
        std::array<VertexPair, 2> edges{};
    };
    std::map<DirectedEdge, BorderCandidates> borderEdges;
    for (auto edge = surface.halfedges_begin(); edge != surface.halfedges_end(); ++edge) {
        if (!edge->is_border()) {
            continue;
        }
        // Use the incident face's orientation, opposite to the border halfedge.
        const auto source = edge->vertex();
        const auto target = edge->opposite()->vertex();
        const auto displacement = target->point() - source->point();
        const auto& canonicalSource = wrappedVertices[source->id()].point;
        const auto& canonicalTarget = wrappedVertices[target->id()].point;

        // Translate both endpoints together so distinct edges with the same
        // wrapped endpoints retain their different lattice displacements.
        const DirectedEdge forward{canonicalSource, canonicalSource + displacement};
        const DirectedEdge reverse{canonicalTarget, canonicalTarget - displacement};
        const std::size_t direction = forward < reverse ? 0 : 1;
        auto& candidates = borderEdges[direction == 0 ? forward : reverse];
        if (candidates.counts[direction]++ == 0) {
            candidates.edges[direction] = {source->id(), target->id()};
        }
    }
    for (const auto& [edge, candidates] : borderEdges) {
        if (candidates.counts[0] != 1 || candidates.counts[1] != 1) {
            result.unmatchedBorderHalfedges += candidates.counts[0] + candidates.counts[1];
            continue;
        }
        const auto [source, target] = candidates.edges[0];
        // The source of the selected copy corresponds to the target of its partner.
        const auto& shift = wrappedVertices[source].latticeShift;
        const auto& partnerShift = wrappedVertices[candidates.edges[1].second].latticeShift;
        result.seamPairs.push_back({source, target, {shift[0] - partnerShift[0],
                                                    shift[1] - partnerShift[1],
                                                    shift[2] - partnerShift[2]}});
    }
    return result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*std::vector<Triangle> extractIntersectingTrianglesOld(P3_Delaunay::Cell_handle cell, Iso_cuboid domain, P3_Delaunay& T, arma::uword iBand, double Ef) {
    std::vector<Triangle> intersectingTriangles;

    double dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();

    // Assuming cell->vertex(i)->info() gives the isosurface function value at vertex i
    std::array<double, 4> values;
    for (int i = 0; i < 4; ++i) {
        arma::vec E = cell->vertex(i)->info();
        values[i] = E(iBand) - Ef;
    }

    // Check for sign changes
    bool isIntersecting = false;

    for (int i = 0; i < values.size() - 1 && !isIntersecting; ++i) {
        for (int j = i + 1; j < values.size(); ++j) {
            if (values[i] * values[j] < 0) {
                isIntersecting = true;
                break; // No need to continue checking once an intersection is found
            }
        }
    }

    if (!isIntersecting) {
        // No intersection or ambiguous case
        return intersectingTriangles;
    }

    // Determine the intersecting edges and compute intersection points
    std::vector<K::Point_3> intersectionPoints;
    //std::vector<P3_Point> intersectionPoints;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (values[i] * values[j] < 0) { // Sign change implies intersection
                // Linear interpolation to find the intersection point
                double ratio = std::fabs(values[i]) / (std::fabs(values[i]) + std::fabs(values[j]));
                PeriodicPoint pp1 = T.periodic_point(cell->vertex(i));
                PeriodicPoint pp2 = T.periodic_point(cell->vertex(j));
                CGAL::Periodic_3_offset_3 offset1 = pp1.second;
                CGAL::Periodic_3_offset_3 offset2 = pp2.second;
                P3_Point p1 = pp1.first;
                P3_Point p2 = pp2.first;

                double x1 = CGAL::to_double(p1.x()) + offset1[0] * dx;
                double y1 = CGAL::to_double(p1.y()) + offset1[1] * dy;
                double z1 = CGAL::to_double(p1.z()) + offset1[2] * dz;

                double x2 = CGAL::to_double(p2.x()) + offset2[0] * dx;
                double y2 = CGAL::to_double(p2.y()) + offset2[1] * dy;
                double z2 = CGAL::to_double(p2.z()) + offset2[2] * dz;

                K::Point_3 actual_point1(x1, y1, z1);
                K::Point_3 actual_point2(x2, y2, z2);
                //P3_Point actual_point1(x1, y1, z1);
                //P3_Point actual_point2(x2, y2, z2);

                K::Point_3 intersection = CGAL::barycenter(actual_point1, 1 - ratio, actual_point2, ratio);
                //P3_Point P3_intersection = intersection;
                //P3_Point P3_intersection = CGAL::barycenter(actual_point1, 1 - ratio, actual_point2, ratio);
                // now we output the offset and see if it is properly handled.

                intersectionPoints.push_back(intersection);
            }
        }
    }

    // Now form triangles from intersection points and orient them
    // First, we record the actual position of vertex(0) in the cell. We will later pass it to another function to orient the triangles.
    PeriodicPoint refpp = T.periodic_point(cell->vertex(0));
    CGAL::Periodic_3_offset_3 offset = refpp.second;
    P3_Point refppp = refpp.first;
    double refX = CGAL::to_double(refppp.x()) + offset[0] * dx;
    double refY = CGAL::to_double(refppp.y()) + offset[1] * dy;
    double refZ = CGAL::to_double(refppp.z()) + offset[2] * dz;
    K::Point_3 referenceVertex{refX, refY, refZ};
    //P3_Point referenceVertex{refX, refY, refZ};
    if (intersectionPoints.size() == 3) { // One triangle
        Triangle tri = {intersectionPoints[0], intersectionPoints[1], intersectionPoints[2]};
        if (!isCorrectlyOriented(tri, cell, domain, values[0], referenceVertex)) {
            std::swap(tri[1], tri[2]); // Swap two vertices to correct orientation
        }
        intersectingTriangles.push_back(tri);
        return intersectingTriangles;
    } else if (intersectionPoints.size() == 4) { // Two triangles
        // Form the first triangle with the first three points
        Triangle tri1 = {intersectionPoints[0], intersectionPoints[1], intersectionPoints[2]};

        // Form the second triangle with the last three points
        Triangle tri2 = {intersectionPoints[1], intersectionPoints[2], intersectionPoints[3]};
        // the order of checking 0-1, 0-2, 0-3, 1-2, 1-3, 2-3 makes sure that the first three and the last three must share a diagonal going through the cell, instead of on one of the four faces.
        if (!isCorrectlyOriented(tri1, cell, domain, values[0], referenceVertex)) {
            std::swap(tri1[1], tri1[2]); // Swap two vertices to correct orientation
        }
        if (!isCorrectlyOriented(tri2, cell, domain, values[0], referenceVertex)) {
            std::swap(tri2[1], tri2[2]); // Swap two vertices to correct orientation
        }

        intersectingTriangles.push_back(tri1);
        intersectingTriangles.push_back(tri2);
        return intersectingTriangles;
    }

    std::cout << "warning! invalid piece of iso surface found.\n";
    return intersectingTriangles;
}*/
///////////////////////
/*std::array<int, 3> calculate_offset(const K_exact::Point_3& p, const Iso_cuboid& domain) {
    std::array<int, 3> offset = {0, 0, 0};
    K::Point_3 point{CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z())};
    auto dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();

    // Calculate offsets in each dimension
    if (point.x() < domain.xmin()) offset[0] = ceil((domain.xmin() - point.x()) / dx);
    else if (point.x() > domain.xmax()) offset[0] = ceil((point.x() - domain.xmax()) / dx) * -1;

    if (point.y() < domain.ymin()) offset[1] = ceil((domain.ymin() - point.y()) / dy);
    else if (point.y() > domain.ymax()) offset[1] = ceil((point.y() - domain.ymax()) / dy) * -1;

    if (point.z() < domain.zmin()) offset[2] = ceil((domain.zmin() - point.z()) / dz);
    else if (point.z() > domain.zmax()) offset[2] = ceil((point.z() - domain.zmax()) / dz) * -1;

    return offset;
}*/
//////////////////////////////
/*K::Point_3 shift_point_into_domain(const K::Point_3& point, const Iso_cuboid& domain, const std::array<int, 3>& offset) {
    double dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();

    // Calculate new coordinates by applying the offset
    double new_x = CGAL::to_double(point.x()) - offset[0] * dx;
    double new_y = CGAL::to_double(point.y()) - offset[1] * dy;
    double new_z = CGAL::to_double(point.z()) - offset[2] * dz;

    // Ensure the point is within the domain boundaries
    new_x = fmod(new_x - domain.xmin(), dx) + domain.xmin();
    new_y = fmod(new_y - domain.ymin(), dy) + domain.ymin();
    new_z = fmod(new_z - domain.zmin(), dz) + domain.zmin();

    // Create and return the shifted point
    return K::Point_3{new_x, new_y, new_z};
}*/
///////////////////////////////////////////
/*double periodicFunction(double fermiLevel, arma::uword bandIndex, Bulk& bulk, K::Point_3& k123)  {
    arma::cx_mat H = bulk.generateLocalHamiltonian(k123[0], k123[1], k123[2]);
    arma::vec eigval = arma::eig_sym(H);

    static unsigned long callCount = 0;
    ++callCount;
    //if (callCount % 1000 == 0)
    //        std::cout << "Hamiltonian called " << callCount << " times, f=" << eigval(bandIndex) - fermiLevel << std::endl;
    return eigval(bandIndex) - fermiLevel;
}*/

/*void refineMeshV1(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef, const Iso_cuboid& domain, double initialEpsilon)   {
    // This function go through all the unique vertices.
    // If one unique cell is found to have the info() of all four vertices close enough to zero, we collect its handle.
    // After that we go through all the collected handles, and insert a P3_Point in the center of the cell, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique cells can be outside the cuboid.
    std::vector<P3_Delaunay::Cell_handle> cellsToRefine; // Assuming this is populated as per your criteria

    // Domain dimensions for shifting points
    auto dx = domain.xmax() - domain.xmin();
    auto dy = domain.ymax() - domain.ymin();
    auto dz = domain.zmax() - domain.zmin();
    // in case it is the 1-cover.
    if (mesh.is_1_cover()) {
        mesh.convert_to_27_sheeted_covering();
        // Now we go through the unique vertices, and copy its info() to all of its images.
        for (auto vit = mesh.vertices_begin(); vit != mesh.vertices_end(); ++vit) {

            auto vh = mesh.get_original_vertex(vit);
            if (vh != vit)
                vit->info() = vh->info();
            //std::cout << count++ << " images info updated\n";

        }
        std::cout << "For interpolation, converted to 27-sheeted covering.\n";
    }

    // Start to refine.
    double epsilon = initialEpsilon;

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {
        // Now go through all unique cells.
        cellsToRefine.clear();
        for (auto cit = mesh.unique_cells_begin(); cit != mesh.unique_cells_end(); ++cit) {
            for (int i = 0; i < 4; ++i) { // Each cell has 4 vertices
                if (fabs(cit->vertex(i)->info()) < fabs(epsilon)) {
                    // If at least one vertex is close enough, mark the cell for refinement
                    cellsToRefine.push_back(cit);
                    break; // No need to check the rest of the vertices for this cell
                }
            }
        }
        // Now all the cells to refine are found, we collect the centroids.
        double maxErr = 0.0;
        std::vector<P3_Point> points_to_insert;
        for (const auto& cell : cellsToRefine) {
            bool allPositive = true, allNegative = true;
            for (int i = 0; i < 4; ++i) {
                if (cell->vertex(i)->info() <= 0) allPositive = false;
                if (cell->vertex(i)->info() >= 0) allNegative = false;
            }

            P3_Point insertPosition;
            if (allPositive || allNegative) {
                // Insert at the centroid
                insertPosition = CGAL::centroid(
                        cell->vertex(0)->point(),
                        cell->vertex(1)->point(),
                        cell->vertex(2)->point(),
                        cell->vertex(3)->point()
                );
            } else {
                // Insert at the weighted position
                double totalWeight = 0.0;
                P3_Point weightedPosition(0, 0, 0);
                for (int i = 0; i < 4; ++i) {
                    auto vertex = cell->vertex(i);
                    double value = vertex->info();
                    if (value*value > maxErr*maxErr)
                        maxErr = std::abs(value);
                    double weight = 1.0 / (std::abs(value) + 1e-8); // Adjust the weight formula as needed
                    weightedPosition = P3_Point(
                            weightedPosition.x() + vertex->point().x() * weight,
                            weightedPosition.y() + vertex->point().y() * weight,
                            weightedPosition.z() + vertex->point().z() * weight
                    );
                    totalWeight += weight;
                }
                // Normalize the weighted position
                insertPosition = P3_Point(
                        weightedPosition.x() / totalWeight,
                        weightedPosition.y() / totalWeight,
                        weightedPosition.z() / totalWeight
                );
            }
            points_to_insert.push_back(insertPosition);
        }
        std::cout << points_to_insert.size() << " centroids found to insert\n";
        // Now we insert all the centroids.

        for (auto & it : points_to_insert)
        {
            P3_Delaunay::Vertex_handle vh = mesh.insert(it);
            double value = periodicFunction(Ef, iBand, bulk, vh->point());
            vh->info() = value;
        }
        epsilon = 2.0*maxErr;
    }





}*/
///////////////////////////////////////
/*void refineMesh(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef)   {
    // This function goes through all the unique edges.
    // If one unique edge is found to be close enough to zero, then collect.
    // After that we go through all the collected unique-edge handles, and insert a P3_Point in the center of the edge, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique edges can be outside the cuboid.
    std::vector<P3_Delaunay::Cell_handle> cellsToRefine; // Assuming this is populated as per your criteria
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    // Domain dimensions for shifting points
    auto dx = domain.xmax() - domain.xmin();
    auto dy = domain.ymax() - domain.ymin();
    auto dz = domain.zmax() - domain.zmin();
    // in case it is the 1-cover.
    if (mesh.is_1_cover()) {
        mesh.convert_to_27_sheeted_covering();
        // Now we go through the unique vertices, and copy its info() to all of its images.
        for (auto vit = mesh.vertices_begin(); vit != mesh.vertices_end(); ++vit) {

            auto vh = mesh.get_original_vertex(vit);
            if (vh != vit)
                vit->info() = vh->info();
            //std::cout << count++ << " images info updated\n";

        }
        std::cout << "For interpolation, converted to 27-sheeted covering.\n";
    }

    // Start to refine.

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {
        // Now go through all unique cells.
        // Collect unique edges that are close enough to zero.
        std::vector<P3_Delaunay::Edge> edges_to_refine;

        for (auto eit = mesh.unique_edges_begin(); eit != mesh.unique_edges_end(); ++eit) {
            auto edge = *eit;
            auto v1 = edge.first->vertex(edge.second);
            auto v2 = edge.first->vertex(edge.third);
            arma::vec E1 = v1->info();
            arma::vec E2 = v2->info();
            double f1 = E1(iBand) - Ef;  // Function value at v1
            double f2 = E2(iBand) - Ef;  // Function value at v2

            // No need to calculate physical edge length or positions in this part,
            // as the refinement decision is based on function values and their linear interpolation

            // Calculate t for the potential zero-crossing point
            double t;
            if (f2 != f1) { // Ensure denominator is not zero
                t = (0 - f1) / (f2 - f1);
            } else {
                // If f1 == f2, we cannot determine a zero-crossing in the traditional sense
                // Skip refinement for this edge, or handle as special case
                continue; // Move to the next edge
            }

            // Check if the zero-crossing is within the edge or within the extended range
            if (-2.0 < t && t < 3.0) {
                // The zero-crossing point is either directly on the edge or close enough for refinement
                edges_to_refine.push_back(edge);
            }
        }


        // Insert middle points on edges close enough to zero.
        double maxErr = 0.0;
        double totalErr = 0.0;
        unsigned long numberOfCrossingEdges = 0;
        std::vector<P3_Point> points_to_insert;
        for (const auto& unique_edge : edges_to_refine) {
            int i = unique_edge.second;
            int j = unique_edge.third;
            P3_Delaunay::Cell_handle cell = unique_edge.first;
            P3_Delaunay::Vertex_handle v1 = cell->vertex(i);
            P3_Delaunay::Vertex_handle v2 = cell->vertex(j);

            arma::vec E1 = v1->info();
            arma::vec E2 = v2->info();
            double f1 = E1(iBand) - Ef;
            double f2 = E2(iBand) - Ef;
            CGAL::Periodic_3_offset_3 offset1 = v1->offset();
            CGAL::Periodic_3_offset_3 offset2 = v2->offset();

            P3_Point canonical_p1 = v1->point();
            P3_Point canonical_p2 = v2->point();

            double x1 = CGAL::to_double(canonical_p1.x()) + offset1[0] * dx;
            double y1 = CGAL::to_double(canonical_p1.y()) + offset1[1] * dy;
            double z1 = CGAL::to_double(canonical_p1.z()) + offset1[2] * dz;

            double x2 = CGAL::to_double(canonical_p2.x()) + offset2[0] * dx;
            double y2 = CGAL::to_double(canonical_p2.y()) + offset2[1] * dy;
            double z2 = CGAL::to_double(canonical_p2.z()) + offset2[2] * dz;


            double t = (0 - f1) / (f2 - f1);
            if (t > 0.0 && t < 1.0)   {
                totalErr += std::abs(f1-f2);
                numberOfCrossingEdges++;
                if (std::abs(f1-f2) > maxErr)
                    maxErr = std::abs(f1-f2);
            }


            // Insert the middle point:
            P3_Point new_point = P3_Point( (x1 + x2)*0.5, (y1+y2)*0.5, (z1+z2)*0.5 );
            P3_Traits traits(domain);
            P3_Point canonical_point = CGAL::P3T3::internal::robust_canonicalize_point(new_point, traits);
            //std::cout << "debug new point: " << canonical_point.x() << " " << canonical_point.y() << " " << canonical_point.z() << "\n";
            points_to_insert.push_back(canonical_point);
        }

        // Now insert all the points, and keep a record of all the new points in a std::set.
        std::unordered_set<P3_Delaunay::Vertex_handle> setOfNewPoints;

        unsigned long count = 0;
        unsigned long totalCall = points_to_insert.size();
        std::cout << "Calculating H for " << points_to_insert.size() << " points" << "\n";
        for (auto & it : points_to_insert)
        {
            P3_Delaunay::Vertex_handle vh = mesh.insert(it);
            setOfNewPoints.insert(vh);
            //double value = periodicFunction(Ef, iBand, bulk, vh->point());
            vh->info() = getEigenValues(bulk, vh->point());
            count++;
            if (count % 1000 == 0)
                std::cout << "Round " << iTimes+1 << ":" << " Hamiltonian evaluated for " << count << "/" << totalCall << " points\n";
        }
        std::cout << "Refinement is done for Round " << iTimes+1 << ", standard div=" << totalErr/(double)numberOfCrossingEdges << ", maxErr=" << maxErr << "\n";
    }
    write_to_vtu_P3("refinedMesh.vtu", mesh, domain, iBand, true, bulk.get_B());
    std::cout << "Refined mesh written in refinedMesh.vtu\n";
}*/
///////////////////////////////////////
/*void refineMeshParallelInherit(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef)   {
    // This function goes through all the unique edges.
    // If one unique edge is found to be close enough to zero, then collect.
    // After that we go through all the collected unique-edge handles, and insert a P3_Point in the center of the edge, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique edges can be outside the cuboid.
    std::vector<P3_Delaunay::Cell_handle> cellsToRefine; // Assuming this is populated as per your criteria
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    // Domain dimensions for shifting points
    auto dx = domain.xmax() - domain.xmin();
    auto dy = domain.ymax() - domain.ymin();
    auto dz = domain.zmax() - domain.zmin();
    // in case it is the 1-cover.
    if (mesh.is_1_cover()) {
        mesh.convert_to_27_sheeted_covering();
        // Now we go through the unique vertices, and copy its info() to all of its images.
        for (auto vit = mesh.vertices_begin(); vit != mesh.vertices_end(); ++vit) {

            auto vh = mesh.get_original_vertex(vit);
            if (vh != vit)
                vit->info() = vh->info();
            //std::cout << count++ << " images info updated\n";

        }
        std::cout << "For interpolation, converted to 27-sheeted covering." << std::endl;
    }

    // Start to refine.

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {
        // Now go through all unique cells.
        // Collect unique edges that are close enough to zero.
        std::vector<P3_Delaunay::Edge> edges_to_refine;

        for (auto eit = mesh.unique_edges_begin(); eit != mesh.unique_edges_end(); ++eit) {
            auto edge = *eit;
            auto v1 = edge.first->vertex(edge.second);
            auto v2 = edge.first->vertex(edge.third);
            arma::vec E1 = v1->info();
            arma::vec E2 = v2->info();
            double f1 = E1(iBand) - Ef;  // Function value at v1
            double f2 = E2(iBand) - Ef;  // Function value at v2

            // No need to calculate physical edge length or positions in this part,
            // as the refinement decision is based on function values and their linear interpolation

            // Calculate t for the potential zero-crossing point
            double t;
            if (f2 != f1) { // Ensure denominator is not zero
                t = (0 - f1) / (f2 - f1);
            } else {
                // If f1 == f2, we cannot determine a zero-crossing in the traditional sense
                // Skip refinement for this edge, or handle as special case
                continue; // Move to the next edge
            }

            // Check if the zero-crossing is within the edge or within the extended range
            if (-2.0 < t && t < 3.0) {
                // The zero-crossing point is either directly on the edge or close enough for refinement
                edges_to_refine.push_back(edge);
            }
        }


        // Insert middle points on edges close enough to zero.
        double maxErr = 0.0;
        double totalErr = 0.0;
        unsigned long numberOfCrossingEdges = 0;
        std::vector<P3_Point> points_to_insert;
        for (const auto& unique_edge : edges_to_refine) {
            int i = unique_edge.second;
            int j = unique_edge.third;
            P3_Delaunay::Cell_handle cell = unique_edge.first;
            P3_Delaunay::Vertex_handle v1 = cell->vertex(i);
            P3_Delaunay::Vertex_handle v2 = cell->vertex(j);

            arma::vec E1 = v1->info();
            arma::vec E2 = v2->info();
            double f1 = E1(iBand) - Ef;
            double f2 = E2(iBand) - Ef;
            CGAL::Periodic_3_offset_3 offset1 = v1->offset();
            CGAL::Periodic_3_offset_3 offset2 = v2->offset();

            P3_Point canonical_p1 = v1->point();
            P3_Point canonical_p2 = v2->point();

            double x1 = CGAL::to_double(canonical_p1.x()) + offset1[0] * dx;
            double y1 = CGAL::to_double(canonical_p1.y()) + offset1[1] * dy;
            double z1 = CGAL::to_double(canonical_p1.z()) + offset1[2] * dz;

            double x2 = CGAL::to_double(canonical_p2.x()) + offset2[0] * dx;
            double y2 = CGAL::to_double(canonical_p2.y()) + offset2[1] * dy;
            double z2 = CGAL::to_double(canonical_p2.z()) + offset2[2] * dz;


            double t = (0 - f1) / (f2 - f1);
            if (t > 0.0 && t < 1.0)   {
                totalErr += std::abs(f1-f2);
                numberOfCrossingEdges++;
                if (std::abs(f1-f2) > maxErr)
                    maxErr = std::abs(f1-f2);
            }


            // Insert the middle point:
            P3_Point new_point = P3_Point( (x1 + x2)*0.5, (y1+y2)*0.5, (z1+z2)*0.5 );
            P3_Traits traits(domain);
            P3_Point canonical_point = CGAL::P3T3::internal::robust_canonicalize_point(new_point, traits);
            //std::cout << "debug new point: " << canonical_point.x() << " " << canonical_point.y() << " " << canonical_point.z() << "\n";
            points_to_insert.push_back(canonical_point);
        }

        // Now, use OpenMP to get the list of Eigenvalues:
        std::vector<arma::vec> listOfEigenValues;
        unsigned long long total = points_to_insert.size();
        listOfEigenValues.resize(total);

        //unsigned long long totalCall = points_to_insert.size();

        unsigned long long completedIterations = 0;

        #pragma omp parallel
        {
            #pragma omp single
            std::cout << "Calculating H for " << points_to_insert.size() << " new points on " << omp_get_num_threads() << " OpenMP threads.\n";
            unsigned long long localCounter = 0; // thread-local counter

            #pragma omp for nowait
            for (unsigned long long i = 0; i < total; i++) {
                listOfEigenValues[i] = getEigenValues(bulk, points_to_insert[i]);
                localCounter++;
                if (localCounter % (total/omp_get_num_threads()/5) == 0)  {
                    #pragma omp atomic
                    completedIterations += localCounter;
                    localCounter = 0; //Reset local counter after updating global counter.

                    #pragma omp critical
                    std::cout << "Round " << iTimes+1 << ":" << " Hamiltonian evaluated for " << completedIterations << "/" << total << " points\n";
                }
            }
        }

        // Now insert the new points and their info();
       // std::unordered_set<P3_Delaunay::Vertex_handle> setOfNewPoints;
        for (unsigned long long i=0; i<points_to_insert.size(); i++)
        {
            P3_Point newPoint = points_to_insert[i];
            P3_Delaunay::Vertex_handle vh = mesh.insert(newPoint);
            //setOfNewPoints.insert(vh);
            vh->info() = listOfEigenValues[i];
        }
        std::cout << "Refinement is done for Round " << iTimes+1 << ", standard div=" << totalErr/(double)numberOfCrossingEdges << ", maxErr=" << maxErr << "\n";
    }
    write_to_vtu_P3("refinedMesh.vtu", mesh, domain, iBand, true, bulk.get_B());
    std::cout << "Refined mesh written in refinedMesh.vtu\n";
}*/
//////////////////////////////
/*void refineMeshParallelBak(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef, int num_OpenMP_threads)   {
    // This function goes through all the unique edges.
    // If one unique edge is found to be close enough to zero, then collect.
    // After that we go through all the collected unique-edge handles, and insert a P3_Point in the center of the edge, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique edges can be outside the cuboid.
    std::vector<P3_Delaunay::Cell_handle> cellsToRefine; // Assuming this is populated as per your criteria
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    // Domain dimensions for shifting points
    auto dx = domain.xmax() - domain.xmin();
    auto dy = domain.ymax() - domain.ymin();
    auto dz = domain.zmax() - domain.zmin();


    // Start to refine.

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {

        // Now go through all unique edges.
        // If a unique edge is close enough to zero, collect its middle point.
        //std::vector<P3_Delaunay::Edge> edges_to_refine;

        std::vector<K::Point_3> points_to_insert;
        double maxErr = 0.0;
        double totalErr = 0.0;
        unsigned long numberOfCrossingEdges = 0;

        for (auto eit = mesh.unique_edges_begin(); eit != mesh.unique_edges_end(); ++eit) {

            auto edge = *eit;
            auto cell = edge.first;
            int i = edge.second;
            int j = edge.third;

            auto v1 = cell->vertex(i);
            auto v2 = cell->vertex(j);

            arma::vec E1 = v1->info().eigVal;
            arma::vec E2 = v2->info().eigVal;
            double f1 = E1(iBand) - Ef;  // Function value at v1
            double f2 = E2(iBand) - Ef;  // Function value at v2



            // No need to calculate physical edge length or positions in this part,
            // as the refinement decision is based on function values and their linear interpolation

            // Calculate t for the potential zero-crossing point
            double t;
            if (f2 != f1) { // Ensure denominator is not zero
                t = (0 - f1) / (f2 - f1);
            } else {
                // If f1 == f2, we cannot determine a zero-crossing in the traditional sense
                // Skip refinement for this edge, or handle as special case
                continue; // Move to the next edge
            }


            // Check if the zero-crossing is within the edge or within the extended range
            if (-2.0 < t && t < 3.0) {
                // The zero-crossing point is either directly on the edge or close enough for refinement
                PeriodicPoint pp1 = mesh.periodic_point(cell, i);
                PeriodicPoint pp2 = mesh.periodic_point(cell, j);
                P3_Delaunay::Offset offset1 = pp1.second;
                P3_Delaunay::Offset offset2 = pp2.second;

                P3_Point canonical_p1 = v1->point();
                P3_Point canonical_p2 = v2->point();

                //if (offset1[0] != 0 || offset1[1] != 0 || offset1[2] != 0 )
                  //  std::cout << "debug offset1 = " << offset1 << std::endl;
                double x1 = CGAL::to_double(canonical_p1.x()) + offset1[0] * dx;
                double y1 = CGAL::to_double(canonical_p1.y()) + offset1[1] * dy;
                double z1 = CGAL::to_double(canonical_p1.z()) + offset1[2] * dz;

                double x2 = CGAL::to_double(canonical_p2.x()) + offset2[0] * dx;
                double y2 = CGAL::to_double(canonical_p2.y()) + offset2[1] * dy;
                double z2 = CGAL::to_double(canonical_p2.z()) + offset2[2] * dz;

                K::Point_3 new_point = P3_Point( (x1 + x2)*0.5, (y1+y2)*0.5, (z1+z2)*0.5 );
                P3_Traits traits(domain);
                K::Point_3 canonical_point = CGAL::P3T3::internal::robust_canonicalize_point(new_point, traits);
                points_to_insert.push_back(canonical_point);
                //points_to_insert.push_back(new_point);
                if (t > 0.0 && t < 1.0)   {
                    totalErr += std::abs(f1-f2);
                    numberOfCrossingEdges++;
                    if (std::abs(f1-f2) > maxErr)
                        maxErr = std::abs(f1-f2);
                }
            }
        }



        // Now, use OpenMP to get the list of Eigenvalues:
        std::vector<arma::vec> listOfEigenValues;
        unsigned long long total = points_to_insert.size();
        for (unsigned long long i=0; i<total; i++)  {
            arma::vec initialVec(bulk.get_matrixN(), arma::fill::zeros);
            listOfEigenValues.push_back(initialVec);
        }


        //unsigned long long totalCall = points_to_insert.size();

        unsigned long long completedIterations = 0;
        omp_set_num_threads(num_OpenMP_threads);

#pragma omp parallel
        {
#pragma omp single
            std::cout << "Calculating H for " << points_to_insert.size() << " new points on " << omp_get_num_threads() << " OpenMP threads." << std::endl;
            unsigned long long localCounter = 0; // thread-local counter

#pragma omp for nowait
            for (unsigned long long i = 0; i < total; i++) {
                auto point = points_to_insert[i];
                arma::cx_mat H = bulk.generateLocalHamiltonian(point.x(), point.y(), point.z());
                listOfEigenValues[i] = arma::eig_sym(H);
                localCounter++;
                if (localCounter % (total/omp_get_num_threads()/5) == 0)  {
#pragma omp atomic
                    completedIterations += localCounter;
                    localCounter = 0; //Reset local counter after updating global counter.
                    if (omp_get_thread_num() == 0) {
#pragma omp critical
                        std::cout << "Round " << iTimes + 1 << ":" << " Hamiltonian evaluated for "
                                  << completedIterations << "/" << total << " points..." << std::endl;
                    }
                }
            }
            if (localCounter > 0) {
                // Now, ensure any unaccounted iterations are added to the global count.
#pragma omp atomic
                completedIterations += localCounter;
            }
        }
        std::cout << "Round " << iTimes + 1 << ":" << " Hamiltonian evaluated for "
                  << completedIterations << "/" << total << " points all done." << std::endl;

        // Now insert the new points and their info();
        // std::unordered_set<P3_Delaunay::Vertex_handle> setOfNewPoints;


        std::cout << "Hamiltonian evaluation finished, now insert points to the mesh...." << std::endl;
        std::vector<P3_Delaunay::Vertex_handle> listOfHandles;

        std::cout << "Before inserting, the mesh has " << mesh.number_of_vertices() << " vertices..." << std::endl;
        for (unsigned long long i=0; i<points_to_insert.size(); i++)
        {
            K::Point_3 newPoint = points_to_insert[i];
            try {

                P3_Delaunay::Vertex_handle vh = mesh.insert(newPoint);
                vh->info().eigVal = listOfEigenValues[i];
                //std::cout << "done.\n";
            }  catch (const std::exception& e)  {
                std::cerr << "An error occured during insertion: " << e.what() << std::endl;
            }
        }
         std::cout << "Refinement is done for Round " << iTimes+1 << ", standard div=" << totalErr/(double)numberOfCrossingEdges << ", maxErr=" << maxErr << std::endl;
    }
    //write_to_vtu_P3("refinedMesh.vtu", mesh, domain, iBand, true, bulk.get_B());
    //std::cout << "Refined mesh written in refinedMesh.vtu\n";
}*/

/////////////////////////////////////////////////////
/*P3_Delaunay initializeMeshV1(Bulk& bulk, const unsigned long num_initial_points, const Iso_cuboid domain) {
    // Now generate a random, periodic delaunay mesh in the cuboid (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5).

    double dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();
    vec b1 = bulk.b1;
    vec b2 = bulk.b2;
    vec b3 = bulk.b3;
    arma::mat B = arma::join_horiz(b1, b2, b3); // This is used to transfer k123 to kxyz coordinates.


    // Randomly insert points in the parallelepiped given by b1 b2 b3.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    P3_Delaunay T(domain);  // Create an empty P3_Delaunay triangulation.

    for (unsigned long i = 0; i < num_initial_points; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        double z = dis(gen);
        P3_Delaunay::Vertex_handle vh = T.insert(P3_Point(x, y, z));// Generate a random point within the box
    }

    // (2) Evaluate the periodic function on all vertices, and assign that as the info() for vertices.
    std::cout << "Initializing " << T.number_of_vertices() << " vertices.\n";
    unsigned long count = 0;
    auto numOfInitialVertices = T.number_of_vertices();
    for (auto vit = T.unique_vertices_begin(); vit != T.unique_vertices_end(); ++vit) {
        //vit->info() = periodicFunction(Ef, iBand, bulk, vit->point());
        vit->info() = getEigenValues(bulk, vit->point());
        count++;
        if (count % 1000 == 0)
            std::cout << count << "/" << numOfInitialVertices << " vertices evaluated.\n";
        //std::cout << "debug the energy is " << vit->info() << "\n";
    }
    write_to_vtu_P3("meshAfterInitialization.vtu", T, domain, 0, true, B);
    return T;
}*/
///////////////////////////////////
/*void refineMeshParallelV1(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef)   {
    // This function goes through all the unique edges.
    // If one unique edge is found to be close enough to zero, then collect.
    // After that we go through all the collected unique-edge handles, and insert a P3_Point in the center of the edge, and then assign its info().
    // Note that we may need to properly handle the periodic boundary because the center of unique edges can be outside the cuboid.
    std::vector<P3_Delaunay::Cell_handle> cellsToRefine; // Assuming this is populated as per your criteria
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    // Domain dimensions for shifting points
    auto dx = domain.xmax() - domain.xmin();
    auto dy = domain.ymax() - domain.ymin();
    auto dz = domain.zmax() - domain.zmin();
    // in case it is the 1-cover.
    if (mesh.is_1_cover()) {
        mesh.convert_to_27_sheeted_covering();
        // Now we go through the unique vertices, and copy its info() to all of its images.
        for (auto vit = mesh.vertices_begin(); vit != mesh.vertices_end(); ++vit) {

            auto vh = mesh.get_original_vertex(vit);
            if (vh != vit)
                vit->info() = vh->info();
            //std::cout << count++ << " images info updated\n";

        }
        std::cout << "For interpolation, converted to 27-sheeted covering.\n";
    }

    // Start to refine.

    for (int iTimes = 0; iTimes < numTimes; iTimes++)   {
        // Now go through all unique cells.
        // Collect unique edges that are close enough to zero.
        std::vector<P3_Delaunay::Edge> edges_to_refine;
        std::set<std::tuple<P3_Delaunay::Vertex_handle, P3_Delaunay::Vertex_handle>> processedEdges;
        for (auto cit = mesh.unique_cells_begin(); cit != mesh.unique_cells_end(); ++cit) {


            // Iterate over all edges of the cell
            for (int i = 0; i < 4; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                    auto v1 = cit->vertex(i);
                    auto v2 = cit->vertex(j);

                    // Consistently order the vertices to uniquely identify the edge
                    auto orderedPair = std::make_tuple(std::min(v1, v2), std::max(v1, v2));

                    // Check if this edge has been processed
                    if (processedEdges.find(orderedPair) == processedEdges.end()) {
                        // Mark this edge as processed
                        processedEdges.insert(orderedPair);

                        // Proceed with your original logic for edge processing
                        arma::vec E1 = v1->info(); // Assuming this returns an arma::vec
                        arma::vec E2 = v2->info(); // Assuming this returns an arma::vec
                        double f1 = E1(iBand) - Ef; // Function value at v1
                        double f2 = E2(iBand) - Ef; // Function value at v2

                        double t;
                        if (f2 != f1) { // Ensure denominator is not zero
                            t = (0 - f1) / (f2 - f1);
                        } else {
                            // If f1 == f2, cannot determine a zero-crossing in the traditional sense
                            continue; // Skip this edge
                        }

                        // Check if the zero-crossing is within the edge or within the extended range
                        if (-2.0 < t && t < 3.0) {
                            // The zero-crossing point is either directly on the edge or close enough for refinement
                            edges_to_refine.emplace_back(P3_Delaunay::Edge(cit, i, j));
                        }
                    }
                }
            }
        }


        // Insert middle points on edges close enough to zero.
        double maxErr = 0.0;
        double totalErr = 0.0;
        unsigned long numberOfCrossingEdges = 0;
        std::vector<P3_Point> points_to_insert;
        for (const auto& unique_edge : edges_to_refine) {
            int i = unique_edge.second;
            int j = unique_edge.third;
            P3_Delaunay::Cell_handle cell = unique_edge.first;
            P3_Delaunay::Vertex_handle v1 = cell->vertex(i);
            P3_Delaunay::Vertex_handle v2 = cell->vertex(j);

            PeriodicPoint pp1 = mesh.periodic_point(v1);
            PeriodicPoint pp2 = mesh.periodic_point(v2);
            CGAL::Periodic_3_offset_3 offset1 = pp1.second;
            CGAL::Periodic_3_offset_3 offset2 = pp2.second;
            P3_Point p1 = pp1.first;
            P3_Point p2 = pp2.first;

            arma::vec E1 = v1->info();
            arma::vec E2 = v2->info();
            double f1 = E1(iBand) - Ef;
            double f2 = E2(iBand) - Ef;
            //CGAL::Periodic_3_offset_3 offset1 = v1->offset();
            //CGAL::Periodic_3_offset_3 offset2 = v2->offset();

            //P3_Point canonical_p1 = v1->point();
            //P3_Point canonical_p2 = v2->point();

            double x1 = CGAL::to_double(p1.x()) + offset1[0] * dx;
            double y1 = CGAL::to_double(p1.y()) + offset1[1] * dy;
            double z1 = CGAL::to_double(p1.z()) + offset1[2] * dz;

            double x2 = CGAL::to_double(p2.x()) + offset2[0] * dx;
            double y2 = CGAL::to_double(p2.y()) + offset2[1] * dy;
            double z2 = CGAL::to_double(p2.z()) + offset2[2] * dz;


            double t = (0 - f1) / (f2 - f1);
            if (t > 0.0 && t < 1.0)   {
                totalErr += std::abs(f1-f2);
                numberOfCrossingEdges++;
                if (std::abs(f1-f2) > maxErr)
                    maxErr = std::abs(f1-f2);
            }


            // Insert the middle point:
            P3_Point new_point = P3_Point( (x1 + x2)*0.5, (y1+y2)*0.5, (z1+z2)*0.5 );
            P3_Traits traits(domain);
            P3_Point canonical_point = CGAL::P3T3::internal::robust_canonicalize_point(new_point, traits);
            //std::cout << "debug new point: " << canonical_point.x() << " " << canonical_point.y() << " " << canonical_point.z() << "\n";
            points_to_insert.push_back(new_point);
        }

        // Now, use OpenMP to get the list of Eigenvalues:
        std::vector<arma::vec> listOfEigenValues;
        unsigned long long total = points_to_insert.size();
        listOfEigenValues.resize(total);

        //unsigned long long totalCall = points_to_insert.size();

        unsigned long long completedIterations = 0;

#pragma omp parallel
        {
#pragma omp single
            std::cout << "Calculating H for " << points_to_insert.size() << " new points on " << omp_get_num_threads() << " OpenMP threads.\n";
            unsigned long long localCounter = 0; // thread-local counter

#pragma omp for nowait
            for (unsigned long long i = 0; i < total; i++) {
                listOfEigenValues[i] = getEigenValues(bulk, points_to_insert[i]);
                localCounter++;
                if (localCounter % (total/omp_get_num_threads()/5) == 0)  {
#pragma omp atomic
                    completedIterations += localCounter;
                    localCounter = 0; //Reset local counter after updating global counter.

#pragma omp critical
                    std::cout << "\rRound " << iTimes+1 << ":" << " Hamiltonian evaluated for " << completedIterations << "/" << total << " points..." << std::flush;
                }
            }
        }
        std::cout << "done." << std::endl;

        // Now insert the new points and their info();
        // std::unordered_set<P3_Delaunay::Vertex_handle> setOfNewPoints;
        for (unsigned long long i=0; i<points_to_insert.size(); i++)
        {
            auto it = points_to_insert[i];
            P3_Delaunay::Vertex_handle vh = mesh.insert(it);
            //setOfNewPoints.insert(vh);
            vh->info() = listOfEigenValues[i];
        }

        std::cout << "Refinement is done for Round " << iTimes+1 << ", standard div=" << totalErr/(double)numberOfCrossingEdges << ", maxErr=" << maxErr << "\n";
    }
    write_to_vtu_P3("refinedMesh.vtu", mesh, domain, iBand, true, bulk.get_B());
    std::cout << "Refined mesh written in refinedMesh.vtu\n";
}*/
//////////////////////////////
/*P3_Delaunay initializeMesh(Bulk& bulk, const unsigned long num_initial_points) {
    // Now generate a random, periodic delaunay mesh in the cuboid (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5).
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    double dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();
    arma::vec3 b1 = bulk.get_b1();
    arma::vec3 b2 = bulk.get_b2();
    arma::vec3 b3 = bulk.get_b3();
    arma::mat33 B = bulk.get_B(); // This is used to transfer k123 to kxyz coordinates.
    arma::mat33 invB = arma::inv(B); // This is to transfer kxyz to k123.
    // Now we find the smallest box that bounds the parallelepiped.
    std::vector<arma::vec3> vertices = {
            -0.5*b1 -0.5*b2 -0.5*b3,
            +0.5*b1 -0.5*b2 -0.5*b3,
            -0.5*b1 +0.5*b2 -0.5*b3,
            -0.5*b1 -0.5*b2 +0.5*b3,
            +0.5*b1 +0.5*b2 -0.5*b3,
            -0.5*b1 +0.5*b2 +0.5*b3,
            +0.5*b1 -0.5*b2 +0.5*b3,
            +0.5*b1 +0.5*b2 +0.5*b3,
    };
    arma::vec3 minVec = vertices[0];
    arma::vec3 maxVec = vertices[0];

// Find min and max for each axis
    for (const auto& vertex : vertices) {
        for (int i = 0; i < 3; ++i) {
            minVec(i) = std::min(minVec(i), vertex(i));
            maxVec(i) = std::max(maxVec(i), vertex(i));
        }
    }

// Now set up the distributions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disX(minVec(0), maxVec(0));
    std::uniform_real_distribution<> disY(minVec(1), maxVec(1));
    std::uniform_real_distribution<> disZ(minVec(2), maxVec(2));

     P3_Delaunay T(domain);  // Create an empty P3_Delaunay triangulation.
    // Now let's generate the points in the bounding box uniformly random.
    // If the points are inside the domain after transferred to 123 coordinate, then include.
    // Otherwise, keep generating.
    //std::fstream debugOutput("debugPoint.txt", std::ios::out);
    unsigned long generatedN = 0;
    while(generatedN < num_initial_points) {
        arma::vec3 randPointXYZ = {disX(gen), disY(gen), disZ(gen)};

        arma::vec3 randPoint123 = invB * randPointXYZ;
        //randPoint123.print("debug 123 = ");
        // Use domain's boundaries for checking
        if (randPoint123(0) > domain.xmin() && randPoint123(0) < domain.xmax() &&
            randPoint123(1) > domain.ymin() && randPoint123(1) < domain.ymax() &&
            randPoint123(2) > domain.zmin() && randPoint123(2) < domain.zmax()) {
            // Insert the point into the Delaunay triangulation
            P3_Delaunay::Vertex_handle vh = T.insert(P3_Point(randPoint123(0), randPoint123(1), randPoint123(2)));
            generatedN++;
    //        debugOutput << randPointXYZ(0) << " " << randPointXYZ(1) << " " << randPointXYZ(2) << " " << 1.0 << "\n";
      //  } else {
      //      debugOutput << randPointXYZ(0) << " " << randPointXYZ(1) << " " << randPointXYZ(2) << " " << 0.0 << "\n";
        }
    }
    //debugOutput.close();
    // Evaluate the eigenvalues on all vertices, and assign that as the info() for vertices.
    std::cout << "Initializing " << T.number_of_vertices() << " vertices.\n";
    unsigned long count = 0;
    auto numOfInitialVertices = T.number_of_vertices();
    for (auto vit = T.unique_vertices_begin(); vit != T.unique_vertices_end(); ++vit) {
        //vit->info() = periodicFunction(Ef, iBand, bulk, vit->point());
        vit->info() = getEigenValues(bulk, vit->point());
        count++;
        if (count % 1000 == 0)
            std::cout << count << "/" << numOfInitialVertices << " vertices evaluated.\n";
        //std::cout << "debug the energy is " << vit->info() << "\n";
    }
    write_to_vtu_P3("meshAfterInitialization.vtu", T, domain, 0, true, B);
    return T;
}*/
/////////////////////////////////////////////////////
/*P3_Delaunay initializeMeshParallelV1(Bulk& bulk, const unsigned long num_initial_points) {
    // Now generate a random, periodic delaunay mesh in the cuboid (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5).
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    double dx = domain.xmax() - domain.xmin();
    double dy = domain.ymax() - domain.ymin();
    double dz = domain.zmax() - domain.zmin();
    arma::vec3 b1 = bulk.get_b1();
    arma::vec3 b2 = bulk.get_b2();
    arma::vec3 b3 = bulk.get_b3();
    arma::mat33 B = bulk.get_B(); // This is used to transfer k123 to kxyz coordinates.
    arma::mat33 invB = arma::inv(B); // This is to transfer kxyz to k123.
    // Now we find the smallest box that bounds the parallelepiped.
    std::vector<arma::vec3> vertices = {
            -0.5*b1 -0.5*b2 -0.5*b3,
            +0.5*b1 -0.5*b2 -0.5*b3,
            -0.5*b1 +0.5*b2 -0.5*b3,
            -0.5*b1 -0.5*b2 +0.5*b3,
            +0.5*b1 +0.5*b2 -0.5*b3,
            -0.5*b1 +0.5*b2 +0.5*b3,
            +0.5*b1 -0.5*b2 +0.5*b3,
            +0.5*b1 +0.5*b2 +0.5*b3,
    };
    arma::vec3 minVec = vertices[0];
    arma::vec3 maxVec = vertices[0];

// Find min and max for each axis
    for (const auto& vertex : vertices) {
        for (int i = 0; i < 3; ++i) {
            minVec(i) = std::min(minVec(i), vertex(i));
            maxVec(i) = std::max(maxVec(i), vertex(i));
        }
    }

// Now set up the distributions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disX(minVec(0), maxVec(0));
    std::uniform_real_distribution<> disY(minVec(1), maxVec(1));
    std::uniform_real_distribution<> disZ(minVec(2), maxVec(2));

    P3_Delaunay T(domain);  // Create an empty P3_Delaunay triangulation.
    // Now let's generate the points in the bounding box uniformly random.
    // If the points are inside the domain after transferred to 123 coordinate, then include.
    // Otherwise, keep generating.
    //std::fstream debugOutput("debugPoint.txt", std::ios::out);
    unsigned long generatedN = 0;
    std::vector<P3_Point> listOfPoints;
    std::vector<P3_Delaunay::Vertex_handle> listOfVertexHandles;
    while(generatedN < num_initial_points) {
        arma::vec3 randPointXYZ = {disX(gen), disY(gen), disZ(gen)};

        arma::vec3 randPoint123 = invB * randPointXYZ;
        //randPoint123.print("debug 123 = ");
        // Use domain's boundaries for checking
        if (randPoint123(0) > domain.xmin() && randPoint123(0) < domain.xmax() &&
            randPoint123(1) > domain.ymin() && randPoint123(1) < domain.ymax() &&
            randPoint123(2) > domain.zmin() && randPoint123(2) < domain.zmax()) {
            // Insert the point into the Delaunay triangulation
            P3_Delaunay::Vertex_handle vh = T.insert(P3_Point(randPoint123(0), randPoint123(1), randPoint123(2)));
            listOfPoints.push_back(vh->point());
            listOfVertexHandles.push_back(vh);
            generatedN++;
            //        debugOutput << randPointXYZ(0) << " " << randPointXYZ(1) << " " << randPointXYZ(2) << " " << 1.0 << "\n";
            //  } else {
            //      debugOutput << randPointXYZ(0) << " " << randPointXYZ(1) << " " << randPointXYZ(2) << " " << 0.0 << "\n";
        }
    }
    //debugOutput.close();
    // Evaluate the eigenvalues on all vertices, and assign that as the info() for vertices.

    unsigned long long completedIterations = 0;
    std::vector<arma::vec> listOfEigenValues;
    listOfEigenValues.resize(listOfPoints.size());
    //omp_set_num_threads(omp_get_max_threads()-4);
    unsigned long long total = listOfPoints.size();
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Initializing " << T.number_of_vertices() << " vertices on " << omp_get_num_threads() << " OpenMP threads.\n";

        unsigned long long localCounter = 0; // thread-local counter

        #pragma omp for nowait
        for (unsigned long long i = 0; i < total; i++) {
            listOfEigenValues[i] = getEigenValues(bulk, listOfPoints[i]);
            localCounter++;
            if (localCounter % (total/omp_get_num_threads()/5) == 0)  {
                #pragma omp atomic
                completedIterations += localCounter;
                localCounter = 0; //Reset local counter after updating global counter.

                #pragma omp critical
                std::cout << completedIterations << "/" << total << " vertices evaluated.\n";
            }

        //std::cout << "debug the energy is " << vit->info() << "\n";
        }

    }
    // Now, write the eigenvalues to the vertices sequentially.
    for (unsigned long long i=0; i<total; i++) {
        auto vit = listOfVertexHandles[i];
        vit->info() = listOfEigenValues[i];
        // debug
        //arma::vec tempDebug = vit->info();
        //std::cout << "tempDebug.size() = " << tempDebug.size() << "\n";
    }
    std::cout << "Initialization of " << T.number_of_vertices() << " vertices finished.\n";
    write_to_vtu_P3("meshAfterInitialization.vtu", T, domain, 0, true, B);
    return T;
}*/

/*Polyhedron extractOneFermiSurfaceMeshV1(Bulk& bulk, arma::uword iBand, double Ef) {

    //refineMesh(T, bulk, iBand, num_refine, Ef);
    // First, make a copy of the mesh.
    std::cout << "Making a copy of the Mesh ..." << std::flush;
    P3_Delaunay T27 = bulk.theMesh;
    std::cout << "done." << std::endl;

    // If is 1-cover, the points outside the cuboid cannot be captured since they can be part of unique edges.
    // We therefore change the mesh to 27-covering.
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    if (T27.is_1_cover()) {
        std::cout << "For interpolation, converting to 27-sheeted covering..." << std::flush;
        T27.convert_to_27_sheeted_covering();
        T27.update_cover_data_after_converting_to_27_sheeted_covering();
        std::cout <<  "done." << std::endl;
    }
    std::vector<K::Point_3> points; // Global list of points (may contain duplicates initially)
    //std::vector<P3_Point> points; // Let's try p3_points and see if it works.
    std::vector<std::vector<std::size_t>> triangles; // Global list of triangles (using indices of points)
    std::cout << "generating surface mesh ..." << std::flush;
    for (auto cl = T27.unique_cells_begin(); cl != T27.unique_cells_end(); ++cl)
    {
        auto trianglesInThisCell = extractIntersectingTriangles(cl, domain, T27, iBand, Ef);
        if (!trianglesInThisCell.empty()) {
            //std::cout << "debug " << trianglesInThisCell.size() << " triangles found.\n";
            for (const auto& tri : trianglesInThisCell) {
                // Insert points from the triangle into the global list
                std::vector<std::size_t> triIndices;
                for (const auto& p : tri) {
                    points.push_back(p); // Add point to global list
                    triIndices.push_back(points.size() - 1); // Use current index for the triangle
                }
                triangles.push_back(triIndices); // Add triangle (indices) to global list
            }
        }
    }
    // At this point, 'points' contains all intersection points (including duplicates),
    // and 'triangles' contains indices of points forming each triangle.

    // (4) Now before we combine the triangles to a soup, we check if any of them are complete outside the cuboid.
    // If so, we shift them back inside the cuboid. This avoid holes on the mesh.

    for (unsigned long i=0; i<triangles.size(); i++)   {
        K::Point_3 p0 = points[triangles[i][0]];
        K::Point_3 p1 = points[triangles[i][1]];
        K::Point_3 p2 = points[triangles[i][2]];
        auto offset0 = calculate_offset(p0, domain);
        auto offset1 = calculate_offset(p1, domain);
        auto offset2 = calculate_offset(p2, domain);
        bool insideCuboid = offset0[0] == offset0[1] == offset0[2] == 0 ||
                            offset1[0] == offset1[1] == offset1[2] == 0 ||
                            offset2[0] == offset2[1] == offset2[2] == 0;
        if (!insideCuboid)
        {
            points[triangles[i][0]] = shift_point_into_domain(p0, domain, offset0);
            points[triangles[i][1]] = shift_point_into_domain(p1, domain, offset1);
            points[triangles[i][2]] = shift_point_into_domain(p2, domain, offset2);
            std::cout << "dangling triangle fixed" << std::endl;
        }
    }
    std::cout << "done." << std::endl;
    // (5) Combine the triangle soup to a Polyhedron_3.
    // Merge duplicate points in the soup
    CGAL::Polygon_mesh_processing::merge_duplicate_points_in_polygon_soup(points, triangles);
    // Then, since triangles are already oriented, we merge the soup to create the Polyhedron_3
    Polyhedron isoSurface;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, triangles, isoSurface);
    return isoSurface;
}*/
//////////////////////////////////
/*Polyhedron extractOneFermiSurfaceMeshTransitionToExact(Bulk& bulk, arma::uword iBand, K_exact::FT Ef) {

    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    std::vector<K_exact::Point_3> points; // Global list of points (may contain duplicates initially)
    std::vector<std::vector<std::size_t>> triangles; // Global list of triangles (using indices of points)
    std::cout << "generating surface mesh ..." << std::flush;
    for (auto cl = bulk.theMesh.unique_cells_begin(); cl != bulk.theMesh.unique_cells_end(); ++cl)
    {
        auto trianglesInThisCell = extractIntersectingTriangles(cl, domain, bulk.theMesh, iBand, Ef);
        if (!trianglesInThisCell.empty()) {
            //std::cout << "debug " << trianglesInThisCell.size() << " triangles found.\n";
            for (const auto& tri : trianglesInThisCell) {
                // Insert points from the triangle into the global list
                std::vector<std::size_t> triIndices;
                for (const auto& p : tri) {
                    points.push_back(p); // Add point to global list
                    triIndices.push_back(points.size() - 1); // Use current index for the triangle
                }
                triangles.push_back(triIndices); // Add triangle (indices) to global list
            }
        }
    }*/
    //std::cout << "debug " << triangles.size()  << " triangles found." << std::endl;
    // At this point, 'points' contains all intersection points (including duplicates),
    // and 'triangles' contains indices of points forming each triangle.

    // (4) Now before we combine the triangles to a soup, we check if any of them are completely outside the cuboid.
    // If so, we shift them back inside the cuboid. This avoids holes on the mesh.
    /*
    for (unsigned long i=0; i<triangles.size(); i++)   {
        K_exact::Point_3 p0 = points[triangles[i][0]];
        K_exact::Point_3 p1 = points[triangles[i][1]];
        K_exact::Point_3 p2 = points[triangles[i][2]];
        auto offset0 = calculate_offset(p0, domain);
        auto offset1 = calculate_offset(p1, domain);
        auto offset2 = calculate_offset(p2, domain);
        bool insideCuboid = offset0[0] == offset0[1] == offset0[2] == 0 ||
                            offset1[0] == offset1[1] == offset1[2] == 0 ||
                            offset2[0] == offset2[1] == offset2[2] == 0;
        if (!insideCuboid)
        {
            points[triangles[i][0]] = shift_point_into_domain(p0, domain, offset0);
            points[triangles[i][1]] = shift_point_into_domain(p1, domain, offset1);
            points[triangles[i][2]] = shift_point_into_domain(p2, domain, offset2);
            std::cout << "dangling triangle fixed" << std::endl;
        }
    }*/

    // (5) Combine the triangle soup to a Polyhedron_3.
    // First, we find the shortest edge length, and set the comparison tolerance to be 0.001 of it.
    /*double min_square_length = std::numeric_limits<double>::max();
    for (const auto& triangle : triangles) {
        // Ensure the triangle has exactly 3 vertices
        if (triangle.size() == 3) {
            // Calculate squared lengths of all edges of the triangle
            K_exact::FT sq_length_01 = CGAL::squared_distance(points[triangle[0]], points[triangle[1]]);
            K_exact::FT sq_length_12 = CGAL::squared_distance(points[triangle[1]], points[triangle[2]]);
            K_exact::FT sq_length_20 = CGAL::squared_distance(points[triangle[2]], points[triangle[0]]);

            // Update min_square_length if any of these squared lengths is smaller
            min_square_length = std::min({min_square_length, sq_length_01, sq_length_12, sq_length_20});
        }
    }*/
    // Now we define the comparison trait:
    /*struct Array_traits {
        double marginSquared; // Member to store the squared margin for comparison

        // Constructor to initialize marginSquared
        Array_traits(double min_squared) : marginSquared(min_squared) {} // Adjusting margin here directly

        struct Equal_3 {
            double marginSquared;
            Equal_3(double marginSquared) : marginSquared(marginSquared) {}

            bool operator()(const K::Point_3& p, const K::Point_3& q) const {
                return CGAL::squared_distance(p, q) < marginSquared;
            }
        };

        struct Less_xyz_3 {
            double marginSquared;
            Less_xyz_3(double marginSquared) : marginSquared(marginSquared) {}

            bool operator()(const K::Point_3& p, const K::Point_3& q) const {
                if (CGAL::squared_distance(p, q) < marginSquared) {
                    return false; // Consider equal within margin, so not less
                }
                // Lexicographical comparison considering margin
                if (std::fabs(p.x() - q.x()) >= marginSquared) {
                    return p.x() < q.x();
                }
                if (std::fabs(p.y() - q.y()) >= marginSquared) {
                    return p.y() < q.y();
                }
                return p.z() < q.z(); // No margin comparison needed for the last coordinate
            }
        };

        Equal_3 equal_3_object() const { return Equal_3(marginSquared); }
        Less_xyz_3 less_xyz_3_object() const { return Less_xyz_3(marginSquared); }
    };*/

    /*std::cout << "done." << std::endl;
    std::cout << "Repairing the triangle soup... " << std::flush;
    CGAL::Polygon_mesh_processing::repair_polygon_soup(points, triangles);
    std::cout << "done." << std::endl;
    // Then, since triangles are already oriented, we merge the soup to create the Polyhedron_3
    Polyhedron isoSurface;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, triangles, isoSurface);

    CGAL::Polygon_mesh_processing::remove_degenerate_edges(isoSurface);
    CGAL::Polygon_mesh_processing::remove_almost_degenerate_faces(isoSurface);

    CGAL::Polygon_mesh_processing::stitch_borders(isoSurface);
    std::cout << "After fixing, we have " << isoSurface.size_of_border_edges() << " edges." << std::endl;

    bool isValid = isoSurface.is_valid();
    if (isValid)
        std::cout << "The surface mesh is valid? " << std::boolalpha << isoSurface.is_valid() << ". " << std::endl;
    else
        std::cout << "Warnning: The surface mesh is not valid." << std::endl;
    return isoSurface;
}*/
