/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef TIGHTBINDINGGEOM_FERMISURFACEEXTRACTION_HPP
#define TIGHTBINDINGGEOM_FERMISURFACEEXTRACTION_HPP

#include "CGAL_dataStructureDefinitions.hpp"
#include <CGAL/Handle_hash_function.h>
#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>
//#include "Bulk.hpp"
class Bulk;
Polyhedron extractOneFermiSurfaceMesh(Bulk& bulk, arma::uword iBand, K_exact::FT Ef);
P3_Delaunay initializeMeshParallel(Bulk& bulk, const unsigned long num_initial_points, int num_OpenMP_threads);
void refineMeshParallel(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef, int num_OpenMP_threads);
void refineMeshParallelForBerryPhaseImplementation(P3_Delaunay& mesh, Bulk& bulk, arma::uword iBand, int numTimes, double Ef, int num_OpenMP_threads, int refineDepth);
K_exact::Point_3 map_to_exact_cuboid(const K_exact::Point_3& p);

using BulkBandEigenstates = std::unordered_map<P3_Delaunay::Vertex_handle, arma::cx_vec,
                                              CGAL::Handle_hash_function>;
void berryCurvatureSingleCell(P3_Delaunay& mesh, P3_Delaunay::Cell_handle& cell,
                              const BulkBandEigenstates& states, arma::vec4& curvature, arma::mat33& basis);
std::vector<P3_Delaunay::Cell_handle> topListOfLargeBerryCurvatures(
    std::vector<P3_Delaunay::Cell_handle>& candidates, P3_Delaunay& mesh,
    const BulkBandEigenstates& states, arma::mat33& basis, int refineDepth);

struct PeriodicSeamPair {
    // One copy of the edge, oriented as part of its incident face.
    std::size_t source;
    std::size_t target;
    // Stored coordinates of this copy minus the corresponding coordinates of its partner.
    std::array<long long, 3> latticeShift;
};

struct PeriodicSurfaceConnectivity {
    std::vector<std::size_t> vertexRepresentative;
    std::vector<PeriodicSeamPair> seamPairs;
    std::size_t unmatchedBorderHalfedges = 0;
};

// The surface's vertices must have contiguous IDs starting at zero.
PeriodicSurfaceConnectivity buildPeriodicSurfaceConnectivity(const Polyhedron& surface);

#endif //TIGHTBINDINGGEOM_FERMISURFACEEXTRACTION_HPP
