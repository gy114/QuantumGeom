/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef TIGHTBINDINGGEOM_CHAIN_HPP
#define TIGHTBINDINGGEOM_CHAIN_HPP

#include <vector>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <armadillo>
#include <stdexcept>
#include <cctype>
#include <unordered_map>
#include <unordered_set>


class Chain {

private:

    std::vector<arma::cx_mat> ListOfH, ListOfOverlap;
    std::vector<int>          ListOfR;
    int matrixN = 0;
    //int nUnits_ = 0;
    //bool canonical_ = false;
    int  rmax_ = 0;
    bool has_half_ = false;
    std::unordered_map<int, size_t> r_index_map_;

    //void parsingOrcaMolden(const char* moldenInputFile, arma::cx_mat& F, arma::cx_mat& S);

public:

    Chain(std::vector<arma::cx_mat> HR, std::vector<int> R);
    Chain(std::vector<arma::cx_mat> HR,
                 std::vector<int> R,
                 std::vector<arma::cx_mat> SR);

    void outputBandStructure(const char* outputFilename, int numSteps);
    arma::cx_mat generateLocalHamiltonian(double k1) const;
    arma::cx_mat generateLocalOverlap(double k1) const;
    arma::cx_mat orthonormalizeHamiltonian(const arma::cx_mat& H,
                                                  const arma::cx_mat& S) const;
    void findExtrema(arma::uword iBand);
    double zakPhase(arma::uword iBand, int N, std::vector<double> &outputZakPhase); // This calculates the zak phase
    double zakPhaseWithS(arma::uword iBand, int N, std::vector<double> &outputZakPhase); // This gives the Zak phase with non-orthogonal basis.
    double zakPhaseNonAbelian(int N, arma::uword iBandMax);


    ///// The following functions are for external access of private members /////////////////////////
    const arma::uword get_matrixN(void);
    ////////////////////////////////////////////  End of external access ////////////////////////////

};


void parseFullMatrix(const std::string& filename, const std::string& marker, arma::mat& M);
void importFockAndOverlap(arma::cx_mat& Fock, arma::cx_mat& S, const std::string& fockFile, const std::string& overlapFile);
void testWithRingOfBenzeneRing(int N);
void testWithITN(int N);
void testImportFockAndOverlap(void);
void testZakPhaseSubSpaceWithCPP(int N);
void testWithITN_AbelianVsNonAbelian(int N, arma::uword iBandMax);


#endif //TIGHTBINDINGGEOM_CHAIN_HPP

