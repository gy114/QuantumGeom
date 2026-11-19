/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef TIGHTBINDINGGEOM_FERMISURFACE_HPP
#define TIGHTBINDINGGEOM_FERMISURFACE_HPP
#include "FermiSurfaceExtraction.hpp"
#include <optional>
#include <vector>
#include <armadillo>

class Bulk;

class FermiSurface {
public:
    struct TrianglePhases {
        arma::uvec bands;
        arma::vec total;
        arma::mat shares;
        arma::vec minSingularValue;
        arma::uvec nearBranchCut;
        arma::vec hallPhase;
        arma::uvec hallUsesConvention;
    };

private:
    struct SheetSubspaceHeader {
        arma::uword band_min = 0;
        arma::uword band_max = 0;

        arma::uword M() const { return band_max - band_min + 1; }
    };

    struct FermiVertexPhysics {
        arma::vec energies;
        arma::cx_mat eigenvectors; // Columns are in energy order within the sheet's band range.
    };

    SheetSubspaceHeader header;
    K_exact::FT Ef = 0; // Example of in-class initialization
    arma::uword iBand = 0; // Initialize to a default value
    std::vector<FermiVertexPhysics> listOfVertexPhysics;
    double degeneracyTolerance = 0.0;
    bool hasSharedBoundary = false;
    bool hallUsesSharedBoundary = false;
    TrianglePhases trianglePhases;

    std::vector<arma::cx_vec> faceEigenstates;
    std::vector<arma::vec3> groupVelocities123; // Use arma::vec3 for a vector of three elements
    std::vector<arma::vec3> spins;  // Use arma::vec3 for a vector of three elements
    std::vector<double> berryCurvature; // Berry Curvature assigned to all faces
    std::vector<double> berryFlux; // Triangle flux accumulated separately on each stored vertex copy.
    std::vector<double> triangleAreasXYZ; // Stores the xyz area of all triangles.
    Polyhedron theSurface;
    PeriodicSurfaceConnectivity periodicConnectivity;
    Iso_cuboid domain;
    void calculateFaceSpins(void);
    arma::mat33 B; // this matrix converts affine to Euclidean coordinate: pxyz = B * p123;
    arma::vec3 b1;
    arma::vec3 b2;
    arma::vec3 b3;
    double calculateEuclideanAreaOneTriangle(Polyhedron::Halfedge_around_facet_circulator hc);
    void calculateEuclideanAreaForAllTriangles(void);
    void prepareContainers();
    arma::vec3 haldaneBoundaryPhases() const;
    SheetSubspaceHeader establishSubspace(Bulk& bulk, const std::vector<arma::vec3>& points,
                                          double gapTolerance, int numThreads) const;
    bool checkSharedBoundary(Bulk& bulk, const std::vector<FermiVertexPhysics>& physics,
                             SheetSubspaceHeader bands, double tolerance) const;

public:


    FermiSurface(double ef, arma::uword iBand);
    void generateSurface(Bulk& bulk);
    void obtainVertexEigenstatesParallel(Bulk& bulk, int num_threads,
                                         std::optional<arma::uword> bandMin = std::nullopt,
                                         std::optional<arma::uword> bandMax = std::nullopt,
                                         std::optional<double> subspaceGap = std::nullopt,
                                         double degeneracyTolerance = 1e-8);
    void obtainFaceEigenstatesAndSpinsParallel(Bulk& bulk, int num_threads);
    void obtainGroupVelocitiesParallelNewton(Bulk& bulk, double eta, int num_threads);
    void obtainGroupVelocitiesParallel(Bulk& bulk, int num_threads);
    void obtainBerryCurvatures(bool sharedBoundary = false);
    const TrianglePhases& wilsonTrianglePhases() const { return trianglePhases; }
    void writeSurfaceVtp(const std::string& filename, bool writeVelocity, bool writeSpin, bool writeBerryCurvature, bool binary);
    void writeBoundaryEdgesVtu(const std::string& filename, bool binary);
    arma::mat33 intrinsicMatrixOfHaldaneIntegral() const;

    arma::mat33 velocityTensor(void);
    arma::mat33 spinVelocityTensor(void);
    void processBorders();

    // Public interface to access and modify the FermiSurface properties
    void setEf(double ef) { this->Ef = K_exact::FT(ef); }
    K_exact::FT getEf() const { return Ef; }
    Iso_cuboid getDomain(void) {return domain; }

    // Additional methods to manipulate the FermiSurface object...
};


#endif //TIGHTBINDINGGEOM_FERMISURFACE_HPP
