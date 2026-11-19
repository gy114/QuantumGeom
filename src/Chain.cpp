/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "Chain.hpp"

#include <algorithm>
#include <complex>
#include <iterator>
#include <limits>
#include <map>
#include <utility>

// Constructor to initialize Chain with HR and R
Chain::Chain(std::vector<arma::cx_mat> HR, std::vector<int> R)
        : ListOfH(std::move(HR)), ListOfR(std::move(R))
{
    if (ListOfH.empty()) {
        matrixN = 0;
        ListOfOverlap.clear();
        return;
    }

    if (ListOfH.size() != ListOfR.size()) {
        throw std::invalid_argument("Sizes of HR and R must match.");
    }

    matrixN = ListOfH[0].n_rows;

    // Initialize all overlaps to zero
    ListOfOverlap.resize(ListOfH.size());
    for (auto &S : ListOfOverlap) {
        S.zeros(matrixN, matrixN);
    }

    // Find the index where R == 0 and set that overlap to identity
    auto it = std::find(ListOfR.begin(), ListOfR.end(), 0);
    if (it == ListOfR.end()) {
        throw std::runtime_error("Overlap initialization: no R=0 term found.");
    }

    std::size_t i0 = std::distance(ListOfR.begin(), it);
    ListOfOverlap[i0].eye(matrixN, matrixN);

    rmax_   = 0;
    for (int r : ListOfR) rmax_ = std::max(rmax_, std::abs(r));
    // populate the unordered map so that we can easily find the match of HR and R.
    for (size_t i = 0; i < ListOfR.size(); ++i) {
        r_index_map_[ListOfR[i]] = i;
    }
}



// helper: enforce Hermiticity
static inline arma::cx_mat symh(const arma::cx_mat& A) {
    return 0.5 * (A + A.t());
}

static inline double frob(const arma::cx_mat& A) {
    return arma::norm(A, "fro");
}

// ===== Constructors =====

Chain::Chain(std::vector<arma::cx_mat> HR,
             std::vector<int> R,
             std::vector<arma::cx_mat> SR)
        : ListOfH(std::move(HR)),
          ListOfR(std::move(R)),
          ListOfOverlap(std::move(SR))
{
    if (ListOfH.empty() ||
        ListOfH.size() != ListOfR.size() ||
        ListOfH.size() != ListOfOverlap.size())
        throw std::invalid_argument("Chain: HR/R/SR size mismatch or empty.");

    matrixN = static_cast<int>(ListOfH[0].n_rows);
    for (size_t k=0; k<ListOfH.size(); ++k) {
        const auto& H = ListOfH[k];
        const auto& S = ListOfOverlap[k];

        if (H.n_rows != H.n_cols || S.n_rows != S.n_cols)
            throw std::invalid_argument("Chain: non-square block at index " + std::to_string(k));
        if (H.n_rows != static_cast<arma::uword>(matrixN) ||
            S.n_rows != static_cast<arma::uword>(matrixN))
            throw std::invalid_argument("Chain: inconsistent block size.");
        if (!H.is_finite() || !S.is_finite())
            throw std::invalid_argument("Chain: non-finite entries.");
    }

    // In full symmetric storage, we simply sum over *all* provided R
    // (we do not *require* the presence of both ±R here).
    //nUnits_ = 0;
    rmax_   = 0;
    for (int r : ListOfR) rmax_ = std::max(rmax_, std::abs(r));
    // populate the unordered map so that we can easily find the match of HR and R.
    for (size_t i = 0; i < ListOfR.size(); ++i) {
        r_index_map_[ListOfR[i]] = i;
    }
}

// ===== k-dependent builders (full symmetric sum) =====
arma::cx_mat Chain::generateLocalHamiltonian(double k1) const {
    arma::cx_mat Hk(matrixN, matrixN, arma::fill::zeros);
    const std::complex<double> I(0.0, 1.0);

    for (int r = -rmax_; r <= rmax_; ++r) {
        if (!r_index_map_.count(r)) continue;

        std::complex<double> phase = std::exp(I * (2.0 * M_PI) * (k1 * double(r)));

        const arma::cx_mat& H_r = ListOfH.at(r_index_map_.at(r));

        if (r == 0 || r == rmax_ || r == -rmax_) {
            // Self-Hermitian block: just add it with phase
            Hk += phase * H_r;
        } else if (r > 0) {
            if (!r_index_map_.count(-r))
                throw std::runtime_error("Missing H(-R) for R = " + std::to_string(r));

            const arma::cx_mat& H_neg_r = ListOfH.at(r_index_map_.at(-r));
            arma::cx_mat H_avg = 0.5 * (H_r + H_neg_r.t());  // symmetrize
            Hk += phase * H_avg;
        } else {
            // r < 0: use the conjugate transpose of the average from +r
            if (!r_index_map_.count(-r))
                throw std::runtime_error("Missing H(-R) for R = " + std::to_string(r));

            const arma::cx_mat& H_pos_r = ListOfH.at(r_index_map_.at(-r));
            const arma::cx_mat& H_neg_r = ListOfH.at(r_index_map_.at(r));
            arma::cx_mat H_avg_dag = 0.5 * (H_pos_r + H_neg_r.t());
            Hk += phase * H_avg_dag.t();  // dagger of average
        }
    }
    arma::cx_mat Hk_herm = 0.5 * (Hk + Hk.t());
    Hk_herm.diag() = arma::cx_vec(arma::real(Hk_herm.diag()), arma::vec(Hk_herm.n_rows, arma::fill::zeros));
    return Hk_herm;
}


/*

static inline double min_eigval_herm(const arma::cx_mat& A) {
    arma::vec d; arma::eig_sym(d, symh(A)); return d.min();
}

// returns (possibly) repaired SPD copy of S, and a flag if a repair was applied
inline std::pair<arma::cx_mat,bool>
ensure_spd_chol(arma::cx_mat S,
                double tau0 = 1e-12,
                double tau_max = 1e-2,
                int max_tries = 20)
{
    arma::cx_mat Sh = symh(S);

    // quick sanity: finite?
    if (!Sh.is_finite()) {
        throw std::runtime_error("ensure_spd_chol: S(k) has non-finite entries");
    }

    // Try raw chol first
    arma::cx_mat L;
    if (arma::chol(L, Sh, "lower")) {
        return {Sh, false}; // already SPD
    }

    // Diagonal loading loop
    double tau = tau0;
    arma::cx_mat I = arma::eye<arma::cx_mat>(Sh.n_rows, Sh.n_cols);
    for (int t = 0; t < max_tries && tau <= tau_max; ++t) {
        arma::cx_mat Stry = Sh + tau * I;
        if (arma::chol(L, Stry, "lower")) {
            return {Stry, true};
        }
        tau *= 2.0;
    }

    // One last try with a larger guard if needed
    arma::cx_mat Sfinal = Sh + tau_max * I;
    if (arma::chol(L, Sfinal, "lower")) {
        return {Sfinal, true};
    }

    throw std::runtime_error("ensure_spd_chol: failed to make S(k) SPD via diagonal loading");
}

*/
arma::cx_mat Chain::generateLocalOverlap(double k1) const {
    arma::cx_mat Sk(matrixN, matrixN, arma::fill::zeros);
    const std::complex<double> I(0.0, 1.0);

    for (int r = -rmax_; r <= rmax_; ++r) {
        if (!r_index_map_.count(r)) continue;

        std::complex<double> phase = std::exp(I * (2.0 * M_PI) * (k1 * double(r)));

        const arma::cx_mat& S_r = ListOfOverlap.at(r_index_map_.at(r));

        if (r == 0 || r == rmax_ || r == -rmax_) {
            Sk += phase * S_r;
        } else if (r > 0) {
            if (!r_index_map_.count(-r))
                throw std::runtime_error("Missing S(-R) for R = " + std::to_string(r));

            const arma::cx_mat& S_neg_r = ListOfOverlap.at(r_index_map_.at(-r));
            arma::cx_mat S_avg = 0.5 * (S_r + S_neg_r.t());
            Sk += phase * S_avg;
        } else {
            if (!r_index_map_.count(-r))
                throw std::runtime_error("Missing S(-R) for R = " + std::to_string(r));

            const arma::cx_mat& S_pos_r = ListOfOverlap.at(r_index_map_.at(-r));
            const arma::cx_mat& S_neg_r = ListOfOverlap.at(r_index_map_.at(r));
            arma::cx_mat S_avg_dag = 0.5 * (S_pos_r + S_neg_r.t());
            Sk += phase * S_avg_dag.t();  // dagger of average
        }
    }
    arma::cx_mat Sk_herm = 0.5 * (Sk + Sk.t());
    Sk_herm.diag() = arma::cx_vec(arma::real(Sk_herm.diag()), arma::vec(Sk_herm.n_rows, arma::fill::zeros));
    return Sk_herm;
}




// ===== band structure writer =====

void Chain::outputBandStructure(const char* outputFilename, int numSteps)
{
    std::ofstream outfile(outputFilename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << outputFilename << std::endl;
        return;
    }
    std::cout << "------ Writing band structure in file " << outputFilename << "------" << std::endl;
    double stepSize = 1.0 / std::max(1, numSteps - 1);
    for (int step = 0; step < numSteps; ++step) {
        double k1 = -0.5 + step * stepSize;

        arma::cx_mat Hk = generateLocalHamiltonian(k1);
        arma::cx_mat Sk = generateLocalOverlap(k1);

        if (!Sk.is_hermitian(1e-8)) {
            std::cerr << "Sk not Hermitian at k = " << k1 << std::endl;
            std::cerr << "Sk - Sk.t(): " << arma::norm(Sk - Sk.t(), "fro") << std::endl;
            Sk.print("Sk = ");
        }

        if (!Hk.is_hermitian(1e-8)) {
            std::cerr << "Hk not Hermitian at k = " << k1 << std::endl;
            std::cerr << "Hk - Hk.t(): " << arma::norm(Sk - Sk.t(), "fro") << std::endl;
            //Hk.print("Hk = ");
        }


        //std::cout << "debug k1 = " << k1 << "    ";

        // Cholesky: Sk = L * L^H
        arma::cx_mat L;
        if (!arma::chol(L, Sk, "lower")) {
            std::cerr << "Cholesky failed at k1 = " << k1 << "\n";
            continue;
        }

        if (!arma::chol(L, 0.5 * (Sk + Sk.t()), "lower")) {
            std::cerr << "[chol fail] at k1 = " << k1 << "\n";
            arma::vec evals;
            if (arma::eig_sym(evals, 0.5 * (Sk + Sk.t()))) {
                std::cerr << "  min eigenvalue of S(k): " << evals.min() << "\n";
            } else {
                std::cerr << "  eig_sym also failed\n";
            }
            continue;
        }



        arma::cx_vec eigval;
        arma::cx_mat eigvec;
        if (!arma::eig_pair(eigval, eigvec, Hk, Sk)) {
            std::cerr << "eig_pair failed at k = " << k1 << "\n";
            continue;
        }
        arma::uvec indices = arma::sort_index(arma::real(eigval));
        arma::cx_vec eigval_sorted = eigval(indices);
        arma::cx_mat eigvec_sorted = eigvec.cols(indices);

        //std::cout << "max |imag(eigval)| = " << arma::abs(arma::imag(eigval)).max() << std::endl;


        outfile << k1;
        for (arma::uword i = 0; i < eigval.n_elem; ++i) outfile << " " << real(eigval_sorted(i));
        outfile << '\n';
    }

    outfile.close();
    std::cout << "------ Band structure written in file " << outputFilename << "------" << std::endl;
}

// ===== small accessor =====
const arma::uword Chain::get_matrixN(void) {
    return static_cast<arma::uword>(matrixN);
}





/*arma::cx_mat Chain::generateLocalHamiltonian(double k1)
{
    using cd = std::complex<double>;
    const cd I(0,1);
    arma::cx_mat Hk(matrixN, matrixN, arma::fill::zeros);

    for (size_t i=0; i<ListOfR.size(); ++i) {
        int r = ListOfR[i];
        const arma::cx_mat& Hr = ListOfH[i];
        if (r == 0) {
            Hk += 0.5 * (Hr + Hr.t());  // ensure Hermitian
        } else if ((r * 2) == *std::max_element(ListOfR.begin(), ListOfR.end()) * 2
                   && std::count(ListOfR.begin(), ListOfR.end(), r) == 1
                   && (r * 2) == (*std::max_element(ListOfR.begin(), ListOfR.end()) * 2)
                   && (ListOfR.back()*2 == static_cast<int>(ListOfR.back())*2) ) {
            // this clause is verbose; simpler rule:
            // if n is even and r == n/2, include as Hermitian (cos term)
            Hk += 0.5 * (Hr + Hr.t());
        } else {
            cd ph = std::exp(cd(0, 2.0*M_PI*k1*r));
            Hk +=  ph * Hr + std::conj(ph) * Hr.t();
        }
    }
    return 0.5*(Hk + Hk.t()); // final clean symmetrization
}
*/

/*
arma::cx_mat Chain::generateLocalOverlap(double k1)
{
    using cd = std::complex<double>;
    const cd I(0,1);
    arma::cx_mat Sk(matrixN, matrixN, arma::fill::zeros);

    for (size_t i=0; i<ListOfR.size(); ++i) {
        int r = ListOfR[i];
        const arma::cx_mat& Sr = ListOfOverlap[i];
        if (r == 0) {
            Sk += 0.5 * (Sr + Sr.t());
        } else if (  false) {
            Sk += 0.5 * (Sr + Sr.t());
        } else {
            cd ph = std::exp(cd(0, 2.0*M_PI*k1*r));
            Sk +=  ph * Sr + std::conj(ph) * Sr.t();
        }
    }
    return 0.5*(Sk + Sk.t());
}

*/


// Function to find extrema of a specific band
void Chain::findExtrema(arma::uword iBand)
{
    if (ListOfH.empty()) {
        std::cerr << "Error: Hamiltonian list is empty." << std::endl;
        return;
    }

    double stepSize = 1.0 / 1000; // Using a finer grid for extrema search
    double maxEnergy = -arma::datum::inf;
    double minEnergy = arma::datum::inf;
    double kAtMax, kAtMin;

    for (double k1 = -0.5; k1 <= 0.5; k1 += stepSize) {
        arma::cx_mat Hk = generateLocalHamiltonian(k1);
        arma::vec eigval;
        arma::eig_sym(eigval, Hk);

        if (iBand < eigval.n_elem) {
            double energy = eigval(iBand);
            if (energy > maxEnergy) {
                maxEnergy = energy;
                kAtMax = k1;
            }
            if (energy < minEnergy) {
                minEnergy = energy;
                kAtMin = k1;
            }
        }
    }

    std::cout << "Band " << iBand << " max energy: " << maxEnergy << " at k = " << kAtMax << std::endl;
    std::cout << "Band " << iBand << " min energy: " << minEnergy << " at k = " << kAtMin << std::endl;
}


// Function to calculate the Zak phase for a given band with N steps
double Chain::zakPhase(arma::uword iBand, int N, std::vector<double> &outputZakPhase)
{
    if (ListOfH.empty()) {
        std::cerr << "Error: Hamiltonian list is empty." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (iBand >= this->matrixN) {
        std::cerr << "Error: band index larger than the basis size." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (N <= 2) {
        std::cerr << "Error: N should be 3 or greater." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    outputZakPhase.clear();  // Clear the vector to store new results
    arma::cx_double totalProduct = arma::cx_double(1.0, 0.0);  // Initialize total product
    std::vector<arma::cx_double> theProduct(iBand + 1, arma::cx_double(1.0, 0.0));  // Product for each band
    std::vector<arma::cx_vec> firstEigenvector;  // Store the first eigenvector
    std::vector<arma::cx_vec> previousEigenvector;  // Store previous eigenvectors

    // Loop over all k1 points, where n = 0 to N-1
    for (int n = 0; n < N; ++n) {
        double k1 = static_cast<double>(n) / N;;

        arma::cx_mat Hk = generateLocalHamiltonian(k1);
        arma::vec eigval;
        arma::cx_mat eigvec;
        arma::eig_sym(eigval, eigvec, Hk);  // Solve the eigenvalue problem, Armadillo sorts eigenvalues

        // If this is the first step, initialize the previous and first eigenvector for each band
        if (n == 0) {
            previousEigenvector.resize(iBand + 1);
            firstEigenvector.resize(iBand + 1);  // Store the first eigenvector for closing the loop later
            for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
                previousEigenvector[bandIndex] = eigvec.col(bandIndex);
                firstEigenvector[bandIndex] = eigvec.col(bandIndex);  // Save the initial vector for the loop closure
            }
            continue;  // Skip to the next iteration after initializing
        }

        // Loop through the bands to compute overlaps
        for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
            // Compute overlap between current and previous eigenvector for the band
            arma::cx_double overlap = arma::cdot(previousEigenvector[bandIndex], eigvec.col(bandIndex));

            // Accumulate the product for each band
            theProduct[bandIndex] *= overlap;

            // Update previous eigenvector for the current band
            previousEigenvector[bandIndex] = eigvec.col(bandIndex);
        }
    }

    // Now close the loop by computing <N-1|0> for each band
    for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
        // Compute the overlap between the last and first eigenvector to close the loop
        arma::cx_double finalOverlap = arma::cdot(previousEigenvector[bandIndex], firstEigenvector[bandIndex]);

        // Multiply the final overlap to complete the product for each band
        theProduct[bandIndex] *= finalOverlap;
    }

    // Now compute the Zak phase for each band
    outputZakPhase.clear();  // Clear the output vector to store results
    for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
        double zakPhaseForBand = std::arg(theProduct[bandIndex]) / M_PI;  // Normalize by π
        outputZakPhase.push_back(zakPhaseForBand);  // Store result for each band

        // Multiply the product across all bands (if required)
        totalProduct *= theProduct[bandIndex];
    }

    // Return the total Zak phase for all bands up to iBand
    double totalZakPhase = std::arg(totalProduct) / M_PI;  // Normalize by π to scale between -1 and 1
    return totalZakPhase;
}






void testWithRingOfBenzeneRing(int N) {
    // Specify the initial value, final value, and number of steps for tPrime
    double tPrimeInitial = 0.5;
    double tPrimeFinal = 3.5;
    int numSteps = 400;

    // Calculate the step size for tPrime
    double tPrimeStep = (tPrimeFinal - tPrimeInitial) / (numSteps - 1);

    // Define the onsite Hamiltonian matrix H(R=0) (constant across all tPrime values)
    arma::cx_mat H0 = {
            {0, -1, 0, 0, 0, -1},
            {-1, 0, -1, 0, 0, 0},
            {0, -1, 0, -1, 0, 0},
            {0, 0, -1, 0, -1, 0},
            {0, 0, 0, -1, 0, -1},
            {-1, 0, 0, 0, -1, 0}
    };

    // Open a file to write the results
    std::ofstream outFile("zak_phase_results.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }

    // Loop over tPrime values
    for (int step = 0; step < numSteps; ++step) {
        double tPrime = tPrimeInitial + step * tPrimeStep;

        // Define the intercell coupling matrix H(R=1) for the current tPrime value
        arma::cx_mat H1 = {
                {0, 0, 0, -tPrime, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0}
        };

        // Compute the Hermitian conjugate of H1 for H(R=-1)
        arma::cx_mat Hm1 = arma::conj(H1.t());  // Use transpose and conjugate for real matrices

        // Collect Hamiltonians and their corresponding Rs
        std::vector<arma::cx_mat> HR = {H0, H1, Hm1};
        std::vector<int> R = {0, 1, -1};

        // Initialize the Chain object with these matrices
        Chain chain(HR, R);
        std::vector<double> zakPhaseList;

        // Compute the Zak phase for the first few bands (e.g., up to band 2) with N=8 steps
        double zakPhase = chain.zakPhase(2, N, zakPhaseList);

        // Write tPrime and the corresponding Zak phase to the file
        outFile << tPrime << "\t" << zakPhase << std::endl;

        // Optionally, print to the console as well
        std::cout << "tPrime = " << tPrime << " -> Total Zak phase: " << zakPhase << std::endl;
    }

    // Close the file after writing
    outFile.close();
    std::cout << "Zak phase results written to 'zak_phase_results.txt'." << std::endl;
}

void testWithITN(int N) {
    // Specify the initial value, final value, and number of steps for tPrime
    double tPrimeInitial = 0.5;
    double tPrimeFinal = 3.5;
    int numSteps = 400;

    double a = 0.0;
    double b = -1.0;

    // Calculate the step size for tPrime
    double tPrimeStep = (tPrimeFinal - tPrimeInitial) / (numSteps - 1);

    // Define the onsite Hamiltonian matrix H(R=0) (constant across all tPrime values)
    arma::cx_mat H0 = {
            {a, b, 0, 0, 0, b, b, 0, 0},
            {b, a, b, 0, 0, 0, 0, b, 0},
            {0, b, a, b, 0, 0, 0, 0, 0},
            {0, 0, b, a, b, 0, 0, 0, 0},
            {0, 0, 0, b, a, b, 0, 0, 0},
            {b, 0, 0, 0, b, a, 0, 0, 0},
            {b, 0, 0, 0, 0, 0, a, 0, 0.69*b},
            {0, b, 0, 0, 0, 0, 0, a, 0.69*b},
            {0, 0, 0, 0, 0, 0, 0.69*b, 0.69*b, a + 1.11*b}
    };

    // Open a file to write the results
    std::ofstream outFile("zak_phase_results.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }

    double tP = 0.88;
    arma::cx_mat H1 = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, tP*b, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    arma::cx_mat Hm1 = arma::conj(H1.t());  // Use transpose and conjugate for real matrices

    // Collect Hamiltonians and their corresponding Rs
    std::vector<arma::cx_mat> HR = {H0, H1, Hm1};
    std::vector<int> R = {0, 1, -1};

    // Initialize the Chain object with these matrices
    Chain testITN(HR, R);
    testITN.outputBandStructure("band_ITN.txt",100);

    // Loop over tPrime values
    for (int step = 0; step < numSteps; ++step) {
        double tPrime = tPrimeInitial + step * tPrimeStep;

        // Define the intercell coupling matrix H(R=1) for the current tPrime value
        arma::cx_mat H1 = {
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, tPrime*b, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0}
        };

        // Compute the Hermitian conjugate of H1 for H(R=-1)
        arma::cx_mat Hm1 = arma::conj(H1.t());  // Use transpose and conjugate for real matrices

        // Collect Hamiltonians and their corresponding Rs
        std::vector<arma::cx_mat> HR = {H0, H1, Hm1};
        std::vector<int> R = {0, 1, -1};

        // Initialize the Chain object with these matrices
        Chain chain(HR, R);
        std::vector<double> zakPhaseList;

        outFile << tPrime;
        for (int uptoNBand = 0; uptoNBand < H0.n_cols; uptoNBand++) {
            // Compute the Zak phase for the first few bands (e.g., up to band 2) with N=8 steps
            double zakPhase = std::abs(chain.zakPhase(uptoNBand, N, zakPhaseList));
            outFile << "\t" << zakPhase;
            std::cout << "uptoNBand=" << uptoNBand << " tPrime = " << tPrime << " -> Total Zak phase: " << zakPhase << std::endl;
        }
        // Write tPrime and the corresponding Zak phase to the file
        outFile << std::endl;

        // Optionally, print to the console as well

    }

    //std::cout << "debug size is " <<H0.size() << std::endl;
    // Close the file after writing
    outFile.close();
    std::cout << "Zak phase results written to 'zak_phase_results.txt'." << std::endl;
}
//////////////////////////////////////////////////////////////
void parseFullMatrix(const std::string& filename, const std::string& marker, arma::mat& M) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    bool in_matrix = false;
    std::vector<int> columnIndices;
    std::map<int, std::map<int, double>> matrixData;  // (row -> (col -> value))
    int maxIndex = -1;
    int debugCount = 0;  // Moved outside loop

    while (std::getline(file, line)) {
        if (!in_matrix) {
            if (line.find(marker) != std::string::npos) {
                in_matrix = true;
            }
            continue;
        }

        if (line.empty()) continue;

        std::istringstream iss_check(line);
        std::string token;
        std::vector<int> tempCols;
        bool isHeader = true;

        while (iss_check >> token) {
            // If token contains '.' or 'e', it's not an integer
            if (token.find('.') != std::string::npos || token.find('e') != std::string::npos) {
                isHeader = false;
                break;
            }
            try {
                tempCols.push_back(std::stoi(token));
            } catch (...) {
                isHeader = false;
                break;
            }
        }


        if (isHeader && tempCols.size() > 1) {
            columnIndices = std::move(tempCols);
            continue;
        }

        //std::cout << "Raw line: [" << line << "]\n";
        std::istringstream iss(line);
        int rowIndex;
        if (!(iss >> rowIndex)) {
            std::cout << "Not interpreted as data line: " << line << "\n";
            continue;
        }

       // std::cout << "Row " << rowIndex << " → ";
        maxIndex = std::max(maxIndex, rowIndex);

        double val;
        for (size_t k = 0; k < columnIndices.size(); ++k) {
            if (!(iss >> val)) {
                std::cout << "[parse fail at k=" << k << "]\n";
                break;
            }
            int colIndex = columnIndices[k];
            maxIndex = std::max(maxIndex, colIndex);
            matrixData[rowIndex][colIndex] = val;
            //std::cout << "(" << rowIndex << "," << colIndex << ")=" << val << " ";
        }
        //std::cout << "\n";

    }

    size_t n = maxIndex + 1;
    M.set_size(n, n);
    M.zeros();

    // Fill matrix, assuming symmetry
    for (const auto& rowPair : matrixData) {
        int i = rowPair.first;
        for (const auto& colPair : rowPair.second) {
            int j = colPair.first;
            M(i, j) = colPair.second;
            M(j, i) = colPair.second;
        }
    }
}

/////////////////////////////////////////////////////////////////////////
void importFockAndOverlap(arma::cx_mat& Fock, arma::cx_mat& S,
                          const std::string& fockFile,
                          const std::string& overlapFile)       {
    arma::mat F_real, S_real;
    parseFullMatrix(fockFile, "Fock matrix for operator 0", F_real);
    parseFullMatrix(overlapFile, "OVERLAP MATRIX", S_real);

    Fock = arma::cx_mat(F_real, arma::mat(F_real.n_rows, F_real.n_cols, arma::fill::zeros));
    S = arma::cx_mat(S_real, arma::mat(S_real.n_rows, S_real.n_cols, arma::fill::zeros));
}
/////////////////////////////////////////////////////////////////////////
void testImportFockAndOverlap(void)   {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)))
        std::cout << "Working directory: " << cwd << std::endl;
    else
        perror("getcwd() error");
    arma::cx_mat F, S;
    importFockAndOverlap(F, S, "last_fock_matrix.txt", "overlap_matrix.txt");
    std::cout << "Fock: " << F.n_rows << " x " << F.n_cols << std::endl;
    std::cout << "S: " << S.n_rows << " x " << S.n_cols << std::endl;
    std::cout << "F(0,0): " << F(0,0) << "\n";
    std::cout << "F(0,1): " << F(0,1) << "\n";
    std::cout << "F(1,0): " << F(1,0) << "\n";
    std::cout << "S(0,0): " << S(0,0) << "\n";
    std::cout << "S(0,1): " << S(0,1) << "\n";
    std::cout << "S(1,0): " << S(1,0) << "\n";
    int N = 960;
    int nUnits = 8;
    int blockSize = N / nUnits;

    std::vector<arma::cx_mat> HR;
    std::vector<arma::cx_mat> SR;
    std::vector<int> R;

    // Loop over r = 0, 1, ..., floor(nUnits/2)
    for (int r = 0; r < nUnits; ++r) {
        int i = 0; // choose a reference unit
        int j = (i + r) % nUnits;
        int row_i = i * blockSize;
        int col_j = j * blockSize;

        arma::cx_mat Hr = F.submat(row_i, col_j, row_i + blockSize - 1, col_j + blockSize - 1);
        arma::cx_mat Sr = S.submat(row_i, col_j, row_i + blockSize - 1, col_j + blockSize - 1);

        int r_mod = (r + nUnits / 2) % nUnits - nUnits / 2;

        if (r_mod == 0) {
            HR.push_back(0.5 * (Hr + Hr.t()));
            SR.push_back(0.5 * (Sr + Sr.t()));
            R.push_back(0);
        } else if (nUnits % 2 == 0 && r_mod == -nUnits / 2) {
            HR.push_back(0.5 * (Hr + Hr.t()));
            SR.push_back(0.5 * (Sr + Sr.t()));
            R.push_back(r_mod);
        } else if (r_mod > 0) {
            HR.push_back(Hr);       SR.push_back(Sr);       R.push_back(+r_mod);
            HR.push_back(Hr.t());   SR.push_back(Sr.t());   R.push_back(-r_mod);
        }
    }


    std::cout << HR.size() << " blocks are detected." << std::endl;
    Chain Chain1_0(HR, R, SR);
    arma::cx_mat H_test = Chain1_0.generateLocalHamiltonian(0.1);
    arma::cx_mat diff = H_test-H_test.t();
    diff.print();
    Chain1_0.outputBandStructure("band_output.txt",100);
    arma::vec spectrum;
    std::cout << "Now try the full spectrum..." << std::endl;
    std::cout << "S is hermisian?: " << S.is_hermitian() << std::endl;
    std::cout << "F is hermisian?: " << F.is_hermitian() << std::endl;
    arma::cx_mat L;
    arma::chol(L, S, "lower");  // S = L^T * L
    arma::cx_mat Linv = arma::inv(L);
    arma::cx_mat F_tilde = Linv.t() * F * Linv;
    arma::eig_sym(spectrum, F_tilde);  // Solve the standard eigenproblem
    spectrum.print("full spectrum: ");

}
/////////////////


arma::cx_mat Chain::orthonormalizeHamiltonian(const arma::cx_mat& H,
                                              const arma::cx_mat& S) const
{
    arma::vec eval;
    arma::cx_mat U;

    if (!arma::eig_sym(eval, U, S)) {
        throw std::runtime_error("eig_sym(S) failed");
    }

    // Build S^{-1/2} = U diag(1/sqrt(lambda)) U†
    arma::cx_mat A = U * arma::diagmat(1.0 / arma::sqrt(eval)) * U.t();

    // Check: A† S A ≈ I
    double err = arma::norm(A.t() * S * A -
                            arma::eye<arma::cx_mat>(S.n_rows, S.n_cols), "fro");
    if (err > 1e-8) {
        std::cerr << "Warning: ||A† S A - I||_F = " << err << std::endl;
    }

    // Transform Hamiltonian: H̃ = A† H A
    return A.t() * H * A;
}

// Function to calculate the Zak phase with non-orthogonal basis.
double Chain::zakPhaseWithS(arma::uword iBand, int N, std::vector<double> &outputZakPhase) {
    if (ListOfH.empty()) {
        std::cerr << "Error: Hamiltonian list is empty." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (iBand >= this->matrixN) {
        std::cerr << "Error: band index larger than the basis size." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (N <= 2) {
        std::cerr << "Error: N should be 3 or greater." << std::endl;
        return std::numeric_limits<double>::quiet_NaN();
    }

    outputZakPhase.clear();  // Clear the vector to store new results
    arma::cx_double totalProduct = arma::cx_double(1.0, 0.0);  // Initialize total product
    std::vector<arma::cx_double> theProduct(iBand + 1, arma::cx_double(1.0, 0.0));  // Product for each band
    std::vector<arma::cx_vec> firstEigenvector;  // Store the first eigenvector
    std::vector<arma::cx_vec> previousEigenvector;  // Store previous eigenvectors

    // Loop over all k1 points, where n = 0 to N-1
    for (int n = 0; n < N; ++n) {
        double k1 = static_cast<double>(n) / N;;
        arma::cx_mat Hk = generateLocalHamiltonian(k1);
        arma::cx_mat Sk = generateLocalOverlap(k1);

// Orthonormalize
        arma::cx_mat Htilde = orthonormalizeHamiltonian(Hk, Sk);

// Diagonalize in orthonormal basis
        arma::vec eigval;
        arma::cx_mat eigvec;
        arma::eig_sym(eigval, eigvec, Htilde);

        // If this is the first step, initialize the previous and first eigenvector for each band
        if (n == 0) {
            previousEigenvector.resize(iBand + 1);
            firstEigenvector.resize(iBand + 1);  // Store the first eigenvector for closing the loop later
            for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
                previousEigenvector[bandIndex] = eigvec.col(bandIndex);
                firstEigenvector[bandIndex] = eigvec.col(bandIndex);  // Save the initial vector for the loop closure
            }
            continue;  // Skip to the next iteration after initializing
        }

        // Loop through the bands to compute overlaps
        for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
            // Compute overlap between current and previous eigenvector for the band
            arma::cx_double overlap = arma::cdot(previousEigenvector[bandIndex], eigvec.col(bandIndex));

            // Accumulate the product for each band
            theProduct[bandIndex] *= overlap;

            // Update previous eigenvector for the current band
            previousEigenvector[bandIndex] = eigvec.col(bandIndex);
        }
    }

    // Now close the loop by computing <N-1|0> for each band
    for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
        // Compute the overlap between the last and first eigenvector to close the loop
        arma::cx_double finalOverlap = arma::cdot(previousEigenvector[bandIndex], firstEigenvector[bandIndex]);

        // Multiply the final overlap to complete the product for each band
        theProduct[bandIndex] *= finalOverlap;
    }

    // Now compute the Zak phase for each band
    outputZakPhase.clear();  // Clear the output vector to store results
    for (arma::uword bandIndex = 0; bandIndex <= iBand; ++bandIndex) {
        double zakPhaseForBand = std::arg(theProduct[bandIndex]) / M_PI;  // Normalize by π
        outputZakPhase.push_back(zakPhaseForBand);  // Store result for each band

        // Multiply the product across all bands (if required)
        totalProduct *= theProduct[bandIndex];
    }

    // Return the total Zak phase for all bands up to iBand
    double totalZakPhase = std::arg(totalProduct) / M_PI;  // Normalize by π to scale between -1 and 1
    return totalZakPhase;
}

double Chain::zakPhaseNonAbelian(int N, arma::uword iBandMax)
{
    if (ListOfH.empty()) { std::cerr << "Error: no H blocks.\n"; return std::numeric_limits<double>::quiet_NaN(); }
    if (iBandMax >= this->matrixN) { std::cerr << "Error: iBandMax >= basis size.\n"; return std::numeric_limits<double>::quiet_NaN(); }
    if (N <= 2) { std::cerr << "Error: N must be ≥ 3.\n"; return std::numeric_limits<double>::quiet_NaN(); }

    const arma::uword M = iBandMax + 1;         // subspace dimension
    arma::cx_mat U_first, U_prev;               // N×M frames
    arma::cx_mat W_total = arma::eye<arma::cx_mat>(M, M);
    bool inited = false;

    for (int n = 0; n < N; ++n) {
        const double k1 = static_cast<double>(n) / N;

        arma::cx_mat Hk = generateLocalHamiltonian(k1);
        arma::cx_mat Sk = generateLocalOverlap(k1);
        arma::cx_mat Htilde = orthonormalizeHamiltonian(Hk, Sk);

        arma::vec eval;
        arma::cx_mat evec;
        if (!arma::eig_sym(eval, evec, Htilde)) {
            std::cerr << "eig_sym(Ĥ) failed at n=" << n << "\n";
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Take the subspace columns (bands 0..iBandMax). Order doesn't matter; SVD link handles mixing.
        arma::cx_mat Uj = evec.cols(0, iBandMax);   // N×M, orthonormal (Euclidean)

        if (!inited) {
            U_first = Uj;
            U_prev  = Uj;
            inited = true;
            continue;
        }

        // Overlap between successive frames (MxM)
        arma::cx_mat O = U_prev.t() * Uj;     // U_prev^† U_j   (in Armadillo .t() works as † in your setup)

        // Polar/unitary part via SVD: O = L Σ R^†  ⇒  W = L R^†  (unitary)
        arma::cx_mat L, R;
        arma::vec s;
        arma::svd(L, s, R, O);
        arma::cx_mat W = L * R.t();

        // Accumulate Wilson loop
        W_total = W_total * W;

        U_prev = std::move(Uj);
    }

    // Close the loop (N-1 → 0)
    {
        arma::cx_mat Oclose = U_prev.t() * U_first;
        arma::cx_mat Lc, Rc; arma::vec sc;
        arma::svd(Lc, sc, Rc, Oclose);
        arma::cx_mat Wclose = Lc * Rc.t();
        W_total = W_total * Wclose;
    }

    // Zak phase of the subspace (units of π)
    arma::cx_double detW = arma::det(W_total);
    return std::arg(detW) / M_PI;
}


void testZakPhaseSubSpaceWithCPP(int N) {
    // Specify the initial value, final value, and number of steps for tPrime
    double tPrimeInitial = 0.5;
    double tPrimeFinal = 3.5;
    int numSteps = 400;

    // Calculate the step size for tPrime
    double tPrimeStep = (tPrimeFinal - tPrimeInitial) / (numSteps - 1);

    // Define the onsite Hamiltonian matrix H(R=0) (constant across all tPrime values)
    arma::cx_mat H0 = {
            {0, -1, 0, 0, 0, -1},
            {-1, 0, -1, 0, 0, 0},
            {0, -1, 0, -1, 0, 0},
            {0, 0, -1, 0, -1, 0},
            {0, 0, 0, -1, 0, -1},
            {-1, 0, 0, 0, -1, 0}
    };

    // Open a file to write the results
    std::ofstream outFile("zak_phase_results.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }

    // Loop over tPrime values
    for (int step = 0; step < numSteps; ++step) {
        double tPrime = tPrimeInitial + step * tPrimeStep;

        // Define the intercell coupling matrix H(R=1) for the current tPrime value
        arma::cx_mat H1 = {
                {0, 0, 0, -tPrime, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0}
        };

        // Compute the Hermitian conjugate of H1 for H(R=-1)
        arma::cx_mat Hm1 = arma::conj(H1.t());  // Use transpose and conjugate for real matrices

        // Collect Hamiltonians and their corresponding Rs
        arma::cx_mat S0(6,6,arma::fill::eye);
        arma::cx_mat S1(6,6,arma::fill::zeros);
        arma::cx_mat Sm1(6,6,arma::fill::zeros);
        std::vector<arma::cx_mat> HR = {H0, H1, Hm1};
        std::vector<arma::cx_mat> SR = {S0, S1, Sm1};
        std::vector<int> R = {0, 1, -1};

        // Initialize the Chain object with these matrices
        Chain chain(HR, R, SR);
        std::vector<double> zakPhaseList;

        // Compute the Zak phase for the first few bands (e.g., up to band 2) with N=8 steps
        double zakPhase = chain.zakPhaseNonAbelian(8, 2);

        // Write tPrime and the corresponding Zak phase to the file
        outFile << tPrime << "\t" << zakPhase << std::endl;

        // Optionally, print to the console as well
        std::cout << "tPrime = " << tPrime << " -> Total Zak phase: " << zakPhase << std::endl;
    }

    // Close the file after writing
    outFile.close();
    std::cout << "Zak phase results written to 'zak_phase_results.txt'." << std::endl;
}
////////////////////////////////////
void testWithITN_AbelianVsNonAbelian(int N, arma::uword iBandMax)
{
    // Scan range for tPrime
    double tPrimeInitial = 0.5;
    double tPrimeFinal   = 3.5;
    int    numSteps      = 400;

    double a = 0.0;
    double b = -1.0;

    double tPrimeStep = (tPrimeFinal - tPrimeInitial) / (numSteps - 1);

    // Onsite Hamiltonian H(R=0)
    arma::cx_mat H0 = {
            {a, b, 0, 0, 0, b, b, 0, 0},
            {b, a, b, 0, 0, 0, 0, b, 0},
            {0, b, a, b, 0, 0, 0, 0, 0},
            {0, 0, b, a, b, 0, 0, 0, 0},
            {0, 0, 0, b, a, b, 0, 0, 0},
            {b, 0, 0, 0, b, a, 0, 0, 0},
            {b, 0, 0, 0, 0, 0, a, 0, 0.69*b},
            {0, b, 0, 0, 0, 0, 0, a, 0.69*b},
            {0, 0, 0, 0, 0, 0, 0.69*b, 0.69*b, a + 1.11*b}
    };

    if (iBandMax >= H0.n_cols) {
        std::cerr << "Error: iBandMax = " << iBandMax
                  << " is >= number of bands = " << H0.n_cols << std::endl;
        return;
    }

    // Open output file: tPrime, Abelian Zak, Non-Abelian Zak
    std::ofstream outFile("zak_phase_abelian_vs_nonabelian.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }

    outFile << "# tPrime\tZak_Abelian(pi)\tZak_NonAbelian(pi)\tDifference\n";

    // (Optional) also output a sample band structure for some reference tPrime
    {
        double tP = 0.88;
        arma::cx_mat H1_ref = arma::zeros<arma::cx_mat>(H0.n_rows, H0.n_cols);
        H1_ref(7, 6) = tP * b;           // same pattern as your ITN coupling
        arma::cx_mat Hm1_ref = arma::conj(H1_ref.t());

        std::vector<arma::cx_mat> HR_ref = {H0, H1_ref, Hm1_ref};
        std::vector<int>          R_ref  = {0, 1, -1};

        Chain testITN(HR_ref, R_ref);
        testITN.outputBandStructure("band_ITN.txt", 100);
    }

    // Scan over tPrime
    for (int step = 0; step < numSteps; ++step) {
        double tPrime = tPrimeInitial + step * tPrimeStep;

        // H(R=1) for this tPrime
        arma::cx_mat H1 = arma::zeros<arma::cx_mat>(H0.n_rows, H0.n_cols);
        H1(7, 6) = tPrime * b;      // same matrix element as before

        arma::cx_mat Hm1 = arma::conj(H1.t());

        std::vector<arma::cx_mat> HR = {H0, H1, Hm1};
        std::vector<int>          R  = {0, 1, -1};

        Chain chain(HR, R);

        std::vector<double> zakPhaseList;

        // Abelian Zak phase of bands 0..iBandMax (in units of π)
        double zakAbelian = std::abs(chain.zakPhase(iBandMax, N, zakPhaseList));

        // Non-Abelian Zak phase of subspace 0..iBandMax (in units of π)
        double zakNonAbelian = std::abs(chain.zakPhaseNonAbelian(N, iBandMax));

        outFile << tPrime << "\t"
                << zakAbelian << "\t"
                << zakNonAbelian << "\t"
                << (zakAbelian - zakNonAbelian) << std::endl;

        std::cout << "tPrime = " << tPrime
                  << "  iBandMax = " << iBandMax
                  << "  Zak_Abelian(pi) = " << zakAbelian
                  << "  Zak_NonAbelian(pi) = " << zakNonAbelian
                  << "  Difference = " << (zakAbelian - zakNonAbelian)
                  << std::endl;
    }

    outFile.close();
    std::cout << "Abelian vs Non-Abelian Zak phases written to 'zak_phase_abelian_vs_nonabelian.txt'."
              << std::endl;
}

