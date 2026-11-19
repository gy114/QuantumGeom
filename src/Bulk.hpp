/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#ifndef MNTEPROJECT_BULK_HPP
#define MNTEPROJECT_BULK_HPP
#define _USE_MATH_DEFINES



#include <vector>
//#include "BandSurfacePiece.hpp"
//#include "FermiCircleInterpolation.hpp"
//#include "CGAL_periodicDataStructureHeader.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <armadillo>
#include "CGAL_dataStructureDefinitions.hpp"
#include "FermiSurfaceExtraction.hpp"
#include "FermiSurface.hpp"

//class FermiSurface;

//using namespace std;
//using namespace arma;
class EigenStates;
class FermiCircleInterpolation;

//typedef complex<double> Complex;

class Bulk {

private:
    arma::vec3 a1;
    arma::vec3 a2;
    arma::vec3 a3;
    arma::vec3 b1;
    arma::vec3 b2;
    arma::vec3 b3;
    arma::uword matrixN;
    arma::mat33 B;
    double spinSplittingEta = 0.0;

    std::vector<arma::vec> ListOfR; // just dimensionless values of R (integers).
    std::vector<arma::vec> ListOfR1R2R3; // affine coordinates of R with units.
    std::vector<arma::cx_mat> ListOfH;
    std::vector<int> ListOfDegeneracyN;
    arma::cx_mat Hamiltonian;
    //std::vector<FermiSurface> listOfFermiSurfaces;

    //// Following variables are for Boltzmann equation. /////
    arma::cx_mat ScatteringMatrixA;
    arma::cx_mat ScatteringMatrixB;
    arma::mat Matrix_S_over_I;
    arma::vec Vec_v_over_I;
    //std::vector<double> listOfMeanFreePath;
    ///// Boltzmann relqted variable end ///////////

    void readFromFile3D(const std::string& wannierHrDatFileName);
    void readUnitCellCart(const std::string& wannierWinFileName, arma::vec3& a1, arma::vec3& a2, arma::vec3& a3);
    void readQuestaalOutput(const std::string& hdf5_filename);

    void GenerateHamiltonian(double k1, double k2, double k3);


public:



    P3_Delaunay theMesh;

    // In case the user does not have the wannier90.win file, then we have to manually input the lattice vectors:
    Bulk(arma::vec& a1, arma::vec& a2, arma::vec& a3, const char* wannierHrDatFileName);
    // If wannier90.win and wannier_hr.dat are both provided, then we can use these files.
    Bulk(const std::string& wannierWinFileName, const std::string& wannierHrDatFileName);
    // Now we implement the initializer for a Bulk using wannier90_tb.dat file.
    Bulk(const std::string& wannierTbFilename);

    void outputBandStructureAlongHighSymmetry(const char* inputFilename, const char* outputFilename, int numStepsFirstLine);
    arma::cx_mat generateLocalHamiltonian(double k1, double k2, double k3); // this does not use Bulk.Hamiltonian to store the outcome, which is OpenMP safe.
    std::vector<arma::cx_mat> generateLocalDHDk(double k1, double k2, double k3); // this returns three components of dH/dk operator.
    arma::vec3 generateLocalVelocity(double k1, double k2, double k3, arma::uword iBand); // this returns a classical vector of velocity for the iband-th eigenstate at k={k1, k2, k3};
    void initializeTheMeshParallel(unsigned long num_of_points, int num_OpenMP_threads);
    void refineTheMeshForOneBandParallel(int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads);
    void refineMeshParallelForBerryPhase(int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads, int depth);
    void writeMeshToVtu(const std::string filename, arma::uword iBand);
    void findExtrema(arma::uword iBand);
    void calculateBerryFluxBZ();
    std::vector<arma::mat33> intrinsicMatrixOfFermiSeaIntegral(
        const std::vector<double>& fermiEnergies, int numThreads = 1);

    ///// The following functions are for Boltzmann equation //////
    void OutputScatteringRateProFit(const char* filename, int indexFullCircle, bool JustA);
    void SolveBTE(bool JustA, double Coefficient, arma::vec FieldDirection, const char* filename); // Coefficient = rho*JH^2*Omega^2/2/pi/hbar;
    void SolveBTEwithCoulomb(double Coefficient, arma::vec FieldDirection, const char* filename, double Q, double RelativeDielectric);
    void SolveBTEwithCoulombIncoherent(double Ratio, double Coefficient, arma::vec FieldDirection, const char* filename, double Q, double RelativeDielectric);
    ///// Boltzmann functions end /////////////////////////////////

    ///// The following functions are for external access of private members /////////////////////////
    arma::vec3 get_b1(void);
    arma::vec3 get_b2(void);
    arma::vec3 get_b3(void);
    arma::mat33 get_B(void);
    arma::cx_mat get_H(void);
    const arma::uword get_matrixN(void);
    void setSmallSpinSplitting(double eta);
    ////////////////////////////////////////////  End of external access ////////////////////////////

    //int N_k1;
    //int N_k2;
    //int N_k3;
    //bool FullBandStructureRecorded;
    //bool FullBandPiecesRecorded;
    //bool FullFermiSurfaceGenerated;
    //bool ScatteringMatricesGenerated;
    //bool CoulombScatteringMatricesGenerated;
    //bool IsKPmodel;
    //std::vector<vec> SpinProfile;
    // Cartisian coordinates of R
   // std::vector<vec> ListOfR1R2;

    //double k1StepRecord;
    //double k2StepRecord;
    //double k3StepRecord;
    //double Alpha;
    //double Beta;
    //double Gamma;
    //double Beta1;
    //double Beta2;
    //double FermiLevel;
    ////////// the following four parameters are for the 2x2 Weyl lattice model;
    //double gamma;
    //double k0;
    //double m;
    //double tx;
    /////////////////
    //cx_mat Gzz;
    //cx_mat Gz;
    //double Delta;
    //double TotalPotentialDrop;
    //double EtaBias;
    // The above two variables are included to resolve degeneracy
    //cx_mat CouplingT;


    //double TotalThickness;
    //std::vector<double> K_OneSide;
    //std::vector<vec> EK_OneSide;
    //std::vector<std::vector<vec> > FullBand; // Used for KP model
    //std::vector<std::vector<vec> > FullBandK;
    //std::vector<std::vector<std::vector<vec> > > FullBand123;  // Used for 3D energy values of the bands in k1,k2,k3
    //std::vector<std::vector<std::vector<vec> > > K123ForTheFullBand;  // The record of the full 3D K values denoted in k1,k2,k3
    //std::vector<FermiCircleInterpolation> ListOfFermiCircle;
    //std::vector<FermiCircleInterpolationKxy> ListOfFermiCircleKxy;
    //std::vector<BandSurfacePiece> ListOfBandSurfacePiece;
    //std::vector<cx_vec> ListOfEigenStatesFullFermiSurface;
    //std::vector<vec> ListOfElectronSpinFullFermiSurface;
    //std::vector<double> ListOfSegmentLengthFullFermiSurface;
    //std::vector<vec> ListOfEnergyGradientFullFermiSurface;
    //std::vector<vec> ListOfMidPointsFullFermiSurface;




    //int BandSurfaceIndexRecord;
    //int NLayers;



    //void GenerateHamiltonian(double k1, double k2);
    //void GenerateHamiltonianKP(double kx, double ky);
    //void GenerateHamiltonianWeyl2x2(double k1, double k2, double k3);
    //void OutputBandStructureKx(double kStart, double kEnd, double kStep);
    //void OutputBandStructureKy(double kStart, double kEnd, double kStep);
    //void OutputBandStructureWeil2x2(const char* filename, int iband);
    //void OutputDOS(const char* filename, double Estart, double Eend, int N_Energy);
    //void CalculateBandStructureOneSide(double kEnd, double kStep);
    //void CalculateFullBand(double kEnd, double kStep, double N_theta);
    //void CalculateFull3DBand(void);
    //void CalculateFullBandK1K2(double k1End, double k2End, int N_k1, int N_k2);
    //void CalculateFullKP_BandKxKy(double kxEnd, double kYEnd, int N_kx, int N_ky);

    //void OutputNthFullBand(const char* filename, int iBand);
    //void GenerateBandSurfaceWithPieces(int iBand);
    //void GenerateBandVolumnWithPieces(int iBand);
    //void OutputFullFermiCircleSpinTextureProFit(const char* filename);
    //void OutputFullFermiCircleSpinTextureProFitKP(const char* filename);
    //void OutputFullFermiCircleGroupVelocityProFit(const char* filename);
    //void OutputBandStructureWeyl2x2(const char *filename, int iband);
    //void OutputGeometryFactor(const char* filename, double E_field_theta);
    //double FindBandMaxium(int iBand, vec & OutputListOfKxyz);
    //void GenerateScatteringMatrix(vec LocalSpinA, double v, double tiltingAngleAlpha);
    //int FindTheStateGoingInThisDirection(vec vXY, int &outputJ_FermiCircle, int &outputI_MidPoint);
    //void SortFermiCirclesIntoFullFermiSurface(void);




    //double CalculateHallAngleFixedEFCoulomb(double Q, double RelativeDielectric,
    //                                        double Coefficient, double E_field_Theta, bool JustA,
    //                                        const char* outputLambdaFilename, double &SpinHallAngle, double& HallAngleUp, double& HallAngleDown);
    //double CalculateHallAngleFixedEFCoulombIncoherent(double Ratio, double Q,
    //                                                  double RelativeDielectric, double Coefficient, double E_field_Theta,
    //                                                  bool JustA, const char* outputLambdaFilename, double& SpinHallAngle);
    //double CalculateHallConductivity(double &OutputSigmaXX, double &OutputSigmaXY, double Coefficient, double E_field_Theta, bool JustA, const char* outputLambdaFilename);
    //double DensityOfStates(double Energy);
    //vec FindFittingParameters(void);
    //void OutputThomasFermiScreeningVector(const char* filename, double RelativeDielectric, double Estart, double Eend, int N_Energy);
    //void OutputMaximumHallAngle(double Ratio, const char* filename, double Coefficient, double Estart, double Eend, double Q, double RelativeDielectric, double StartThetaDeg, double EndThetaDeg, int N_theta, int N_energy);
};


#endif //MNTEPROJECT_BULK_HPP
