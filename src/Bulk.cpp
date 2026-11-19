/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "Bulk.hpp"
#include "CGAL_dataStructureDefinitions.hpp"
#include "FermiSurfaceExtraction.hpp"
#include "WriteToVTK.hpp"
#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>



Bulk::Bulk(arma::vec& a1, arma::vec& a2, arma::vec& a3, const char* wannierHrDatFileName)
{
    this->a1 = a1;
    this->a2 = a2;
    this->a3 = a3;
    // Calculate the volume of the unit cell
    double volume = dot(this->a1, cross(this->a2, this->a3));
    // Calculate reciprocal lattice vectors
    arma::vec new_b1 = 2 * arma::datum::pi * cross(a2, a3) / volume;
    arma::vec new_b2 = 2 * arma::datum::pi * cross(a3, a1) / volume;
    arma::vec new_b3 = 2 * arma::datum::pi * cross(a1, a2) / volume;
    this->b1 = new_b1;
    this->b2 = new_b2;
    this->b3 = new_b3;
    this->B = arma::join_horiz(new_b1, new_b2, new_b3);
    this->readFromFile3D(wannierHrDatFileName);
    writeParallelepipedEdgesVtu(this->b1, this->b2, this->b3, "BrillouinZone.vtu");
    std::cout << "Brillouin zone (parallelepiped) written to file BrillouinZone.vtu" << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::readUnitCellCart(const std::string& wannierWinFileName, arma::vec3& a1, arma::vec3& a2, arma::vec3& a3) {
    std::ifstream file(wannierWinFileName);
    if (!file) {
        std::cerr << "Unable to open file: " << wannierWinFileName << std::endl;
        return;
    }

    std::string line;
    bool read_section = false;
    int vectorCounter = 0; // Initialize the counter to keep track of the vectors

    while (std::getline(file, line)) {
        if (line.find("begin unit_cell_cart") != std::string::npos) {
            read_section = true;
            continue;
        }
        if (line.find("end unit_cell_cart") != std::string::npos) {
            break; // Exit the loop once the end of the unit cell section is reached
        }
        if (read_section) {
            std::istringstream iss(line);
            arma::vec3 tempVector;
            if (iss >> tempVector(0) >> tempVector(1) >> tempVector(2)) {
                switch (vectorCounter) {
                    case 0:
                        a1 = tempVector;
                        break;
                    case 1:
                        a2 = tempVector;
                        break;
                    case 2:
                        a3 = tempVector;
                        break;
                }
                vectorCounter++; // Increment the counter after assigning the values
            }
        }
    }

    if (vectorCounter < 3) {
        std::cerr << "Warning: Not all unit cell vectors were read successfully." << std::endl;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
Bulk::Bulk(const std::string& wannierWinFileName, const std::string& wannierHrDatFileName)  {
    this->readFromFile3D(wannierHrDatFileName);
    arma::vec3 new_a1, new_a2, new_a3;
    this->readUnitCellCart(wannierWinFileName, new_a1, new_a2, new_a3);
    this->a1 = new_a1;
    this->a2 = new_a2;
    this->a3 = new_a3;
    // Calculate the volume of the unit cell

    std::cout << "============ Rows of direct lattice vectors are: =================" << std::endl;
    this->a1.t().print();
    this->a2.t().print();
    this->a3.t().print();
    std::cout << "==================================================================\n\n\n" << std::endl;


    double volume = dot(this->a1, cross(this->a2, this->a3));
    //std::cout << "debug volume " << volume << "\n";
    // Calculate reciprocal lattice vectors++
    arma::vec3 new_b1 = 2 * arma::datum::pi * cross(a2, a3) / volume;
    arma::vec3 new_b2 = 2 * arma::datum::pi * cross(a3, a1) / volume;
    arma::vec3 new_b3 = 2 * arma::datum::pi * cross(a1, a2) / volume;
    this->b1 = new_b1;
    this->b2 = new_b2;
    this->b3 = new_b3;
    this->B = arma::join_horiz(new_b1, new_b2, new_b3);
    // Output the reciprocal lattice vectors
    std::cout << "============ Rows of reciprocal lattice vectors are: =============" << std::endl;
    this->b1.t().print();
    this->b2.t().print();
    this->b3.t().print();
    std::cout << "==================================================================\n\n\n" << std::endl;
    writeParallelepipedEdgesVtu(this->b1, this->b2, this->b3, "BrillouinZone.vtu");
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
Bulk::Bulk(const std::string& wannierTbFileName) {
    std::cout << "======== Reading tight-binding model.. =========\n";
    std::ifstream inputFile(wannierTbFileName);
    std::string line;

    this->ListOfR.clear();
    this->ListOfH.clear();
    this->ListOfDegeneracyN.clear();
    this->ListOfR1R2R3.clear();

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }

    // Reading the title line
    getline(inputFile, line);
    std::string titleLine = line; // Storing the timestamp
    std::cout << "Timestamp: " << titleLine << std::endl;

    // Read the lattice vectors
    getline(inputFile, line);
    std::istringstream issLatticeVector1(line);
    double ax, ay, az;
    issLatticeVector1 >> ax >> ay >> az;
    this->a1 = arma::vec3{ax, ay, az};
    getline(inputFile, line);
    std::istringstream issLatticeVector2(line);
    issLatticeVector2 >> ax >> ay >> az;
    this->a2 = arma::vec3{ax, ay, az};
    getline(inputFile, line);
    std::istringstream issLatticeVector3(line);
    issLatticeVector3 >> ax >> ay >> az;
    this->a3 = arma::vec3{ax, ay, az};
    std::cout << "============ Rows of direct lattice vectors are: =================" << std::endl;
    this->a1.t().print();
    this->a2.t().print();
    this->a3.t().print();
    std::cout << "==================================================================\n" << std::endl;

    // Calculate the volume of the unit cell
    double volume = dot(this->a1, cross(this->a2, this->a3));
    // Calculate reciprocal lattice vectors
    arma::vec new_b1 = 2 * arma::datum::pi * cross(a2, a3) / volume;
    arma::vec new_b2 = 2 * arma::datum::pi * cross(a3, a1) / volume;
    arma::vec new_b3 = 2 * arma::datum::pi * cross(a1, a2) / volume;
    this->b1 = new_b1;
    this->b2 = new_b2;
    this->b3 = new_b3;
    this->B = arma::join_horiz(new_b1, new_b2, new_b3);


    // Reading the first integer value
    getline(inputFile, line);
    int matrixN;
    std::istringstream iss1(line);
    if (!(iss1 >> matrixN)) {
        std::cerr << "Error reading the matrix size." << std::endl;
        return;
    }
    this->matrixN = matrixN;
    this->Hamiltonian.set_size(matrixN, matrixN);
    this->Hamiltonian.zeros();

    // Reading the second integer value
    getline(inputFile, line);
    int sizeOfR;
    std::istringstream iss2(line);
    if (!(iss2 >> sizeOfR)) {
        std::cerr << "Error reading the size of {R}" << std::endl;
        return;
    }


    // Read the degeneracy data
    for (int i = 0; i < sizeOfR && getline(inputFile, line); ) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        int n;
        while (iss >> n) {
            this->ListOfDegeneracyN.push_back(n);
            i++; // increment for each degeneracy data read
        }
    }


    // Now read the matrix data
    arma::cx_mat H_iterator(matrixN, matrixN, arma::fill::zeros);
    arma::vec R_iterator(3);
    double R1, R2, R3;
    int I, J;


    while (getline(inputFile, line)) {
        if (line.empty())
            continue;

        std::istringstream iss(line);
        if (iss >> R1 >> R2 >> R3) {
            R_iterator = R1*a1 + R2*a2 + R3*a3;
            arma::vec3 R_temp = {R1, R2, R3};
            bool validBlock = true;
            while (getline(inputFile, line) && !line.empty()) {
                std::istringstream issMatrix(line);
                std::vector<double> nums;
                double temp;
                while (issMatrix >> temp) {
                    nums.push_back(temp);
                }

                if (nums.size() == 4) { // It is a valid matrix line
                    I = static_cast<int>(nums[0]);
                    J = static_cast<int>(nums[1]);
                    std::complex<double> element(nums[2], nums[3]);
                    H_iterator(I - 1, J - 1) = element;
                } else if (nums.size() == 8) {
                    validBlock = false;
                    break;
                }
            }
            if (validBlock) {
                this->ListOfR.push_back(R_iterator);
                this->ListOfR1R2R3.push_back(R_temp);
                this->ListOfH.push_back(H_iterator);
            } else
                break;
        }
    }

    std::cout << "Size of Matrix (matrixN): " << this->matrixN << std::endl;
    std::cout << "Size of {ListOfDegeneracyN}: " << this->ListOfDegeneracyN.size() << std::endl;
    std::cout << "The targeted size of {R}: " << sizeOfR << std::endl;
    std::cout << this->ListOfH.size() << " matrices are imported." << std::endl;
    std::cout << "The actual size of {R} is " << this->ListOfR.size() << std::endl;
    writeParallelepipedEdgesVtu(new_b1, new_b2, new_b3, "BrillouinZone.vtu");
    std::cout << "Brillouin zone (parallelepiped) written to file BrillouinZone.vtu" << std::endl;
    std::cout << "======== Reading completed             =========\n\n\n";

    inputFile.close();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::readFromFile3D(const std::string& wannierHrDatFileName) {

    std::cout << "======== Reading tight-binding model.. =========\n";
    std::ifstream inputFile(wannierHrDatFileName);
    std::string line;

    this->ListOfR.clear();
    this->ListOfH.clear();
    this->ListOfDegeneracyN.clear();
    this->ListOfR1R2R3.clear();

    if (!inputFile.is_open()) {
            std::cerr << "Error opening file" << std::endl;
            return;
    }

    // Reading the title line
    getline(inputFile, line);
    std::string titleLine = line; // Storing the timestamp

    // Reading the first integer value
    getline(inputFile, line);
    int matrixN;
    std::istringstream iss1(line);
    if (!(iss1 >> matrixN)) {
            std::cerr << "Error reading the matrix size." << std::endl;
            return;
    }
    this->matrixN = matrixN;
    this->Hamiltonian.set_size(matrixN, matrixN);
    this->Hamiltonian.zeros();

    // Reading the second integer value
    getline(inputFile, line);
    int sizeOfR;
    std::istringstream iss2(line);
    if (!(iss2 >> sizeOfR)) {
            std::cerr << "Error reading the size of {R}" << std::endl;
            return;
    }

    arma::cx_mat H_iterator(matrixN, matrixN, arma::fill::zeros);
    arma::vec R_iterator(3);
    double R1, R2, R3;
    int I, J;
    double RealPart, ImagPart;


    // Read the degeneracy data
    for (int i = 0; i < sizeOfR && getline(inputFile, line); ) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        int n;
        while (iss >> n) {
            this->ListOfDegeneracyN.push_back(n);
            i++; // increment for each degeneracy data read
        }
    }

    // Now read the matrix data
    while (getline(inputFile, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        iss >> R1 >> R2 >> R3 >> I >> J >> RealPart >> ImagPart;

        // Process the first line of the matrix
        R_iterator = R1 * a1 + R2 * a2 + R3 * a3;
        if (I == 1 && J == 1) {
            this->ListOfR.push_back(R_iterator);
            arma::vec3 temp = {R1, R2, R3};
            this->ListOfR1R2R3.push_back(temp);
        }
        H_iterator(I - 1, J - 1) = std::complex<double>(RealPart, ImagPart);

        // Read the remaining elements of the matrix
        for (int i = 1; i < matrixN * matrixN; ++i) {
            if (!getline(inputFile, line)) break;
            std::istringstream issMatrix(line);
            issMatrix >> R1 >> R2 >> R3 >> I >> J >> RealPart >> ImagPart;
            H_iterator(I - 1, J - 1) = std::complex<double>(RealPart, ImagPart);
        }
        this->ListOfH.push_back(H_iterator);
        H_iterator.fill(0); // Reset the matrix for the next iteration
    }


    std::cout << "Title line: " << titleLine << std::endl;
    std::cout << "Size of Matrix (matrixN): " << this->matrixN << std::endl;
    std::cout << "The targeted size of {R}: " << sizeOfR << std::endl;
    std::cout << this->ListOfH.size() << " matrices are imported." << std::endl;
    std::cout << "The actual size of {R} is " << this->ListOfR.size() << std::endl;
    std::cout << "======== Reading completed             =========\n\n\n";
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::readQuestaalOutput(const std::string &hdf5_filename) {

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::GenerateHamiltonian(double k1, double k2, double k3) {

    arma::vec3 k;
    arma::vec3 k1k2k3;
    k1k2k3 = {k1, k2, k3};
    this->Hamiltonian.zeros();
    // Preallocate memory for phase * ListOfH[i] to avoid repeated allocations
    arma::cx_mat tempHamiltonian = arma::zeros<arma::cx_mat>(this->Hamiltonian.n_rows, this->Hamiltonian.n_cols);

    for (int i=0; i<this->ListOfR.size(); i++)
    {
        std::complex<double> phase = exp(2.0 * M_PI * std::complex<double>(0.0, 1.0) * dot(this->ListOfR1R2R3[i], k1k2k3)) / static_cast<double>(this->ListOfDegeneracyN[i]);
        tempHamiltonian = this->ListOfH[i] * phase;
        this->Hamiltonian += tempHamiltonian;
    }
    /// Now we add a little spin splitting.
    arma::uword N = this->ListOfH[0].n_rows;
    arma::cx_mat I;
    I.eye(N/2, N/2);
    arma::cx_mat sigmaZ = arma::cx_mat{{std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)},
                                       {std::complex<double>(0.0, 0.0), std::complex<double>(-1.0, 0.0)}};

    arma::cx_mat sigmaX = arma::cx_mat{{std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0)},
                                       {std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)}};

    arma::cx_mat sigmaY = arma::cx_mat{{std::complex<double>(0.0, 0.0), std::complex<double>(0.0, -1.0)},
                                       {std::complex<double>(0.0, 1.0), std::complex<double>(0.0, 0.0)}};

    arma::cx_mat SigmaX, SigmaY, SigmaZ;
    SigmaZ = kron(sigmaZ, I);
    SigmaX = kron(sigmaX, I);
    SigmaY = kron(sigmaY, I);
    arma::vec3 kxyz = this->b1*k1 + this->b2*k2 + this->b3*k3;
    arma::cx_mat hSplitting = -SigmaZ*0.0000001;// - 0.001*SigmaX*kxyz(0) - 0.001*SigmaY*kxyz(1);
    this->Hamiltonian = this->Hamiltonian + hSplitting;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::cx_mat Bulk::generateLocalHamiltonian(double k1, double k2, double k3)    {
    arma::cx_mat H(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat tempHamiltonian(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::vec3 k;
    arma::vec3 k1k2k3;
    k1k2k3 = {k1, k2, k3};
    for (int i=0; i<this->ListOfR.size(); i++)
    {
        std::complex<double> phase = exp(2.0 * M_PI * std::complex<double>(0.0, 1.0) * dot(this->ListOfR1R2R3[i], k1k2k3)) / static_cast<double>(this->ListOfDegeneracyN[i]);
        tempHamiltonian = this->ListOfH[i] * phase;
        H += tempHamiltonian;
    }
    /// Now we add a little spin splitting if N is even.
    arma::uword N = this->ListOfH[0].n_rows;
    if (N%2 == 0) {
        arma::cx_mat I;
        I.eye(N/2, N/2);
        arma::cx_mat sigmaZ = arma::cx_mat{{std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)},
                                       {std::complex<double>(0.0, 0.0), std::complex<double>(-1.0, 0.0)}};
        arma::cx_mat SigmaX, SigmaY, SigmaZ;
        SigmaZ = kron(sigmaZ, I);

        arma::cx_mat hSplitting = -SigmaZ*this->spinSplittingEta;
        H = H + hSplitting;
    }
    return H;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<arma::cx_mat> Bulk::generateLocalDHDk(double k1, double k2, double k3)    {
    std::vector<arma::cx_mat> DHDk123;
    arma::cx_mat DHDk0(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat DHDk1(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat DHDk2(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat tempMatrix;
    arma::vec3 k1k2k3 = arma::vec3{k1, k2, k3};
    for (int i=0; i<this->ListOfR.size(); i++)
    {
        arma::cx_vec phase = 2.0 * M_PI * std::complex<double>(0.0, 1.0) * this->ListOfR1R2R3[i] *
                                   exp(2.0 * M_PI * std::complex<double>(0.0, 1.0) * dot(this->ListOfR1R2R3[i], k1k2k3))
                                   / static_cast<double>(this->ListOfDegeneracyN[i]);
        DHDk0 += phase(0) * this->ListOfH[i];
        DHDk1 += phase(1) * this->ListOfH[i];
        DHDk2 += phase(2) * this->ListOfH[i];
    }
    DHDk123.push_back(DHDk0);
    DHDk123.push_back(DHDk1);
    DHDk123.push_back(DHDk2);
    return DHDk123;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::vec3 Bulk::generateLocalVelocity(double k1, double k2, double k3, arma::uword iBand) {
    arma::cx_mat DHDk0(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat DHDk1(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat DHDk2(this->matrixN, this->matrixN, arma::fill::zeros);
    arma::cx_mat tempMatrix;
    arma::vec3 k1k2k3 = arma::vec3{k1, k2, k3};
    for (int i=0; i<this->ListOfR.size(); i++)
    {
        arma::cx_vec phase = 2.0 * M_PI * std::complex<double>(0.0, 1.0) * this->ListOfR1R2R3[i] *
                             exp(2.0 * M_PI * std::complex<double>(0.0, 1.0) * dot(this->ListOfR1R2R3[i], k1k2k3))
                             / static_cast<double>(this->ListOfDegeneracyN[i]);
        DHDk0 += phase(0) * this->ListOfH[i];
        DHDk1 += phase(1) * this->ListOfH[i];
        DHDk2 += phase(2) * this->ListOfH[i];
    }
    arma::vec eigval;
    arma::cx_mat eigvec, H;
    H = this->generateLocalHamiltonian(k1, k2, k3);
    arma::eig_sym(eigval, eigvec, H);
    arma::cx_vec psi = eigvec.col(iBand);
    arma::cx_mat v0 = ((psi.t())*DHDk0*psi);
    arma::cx_mat v1 = ((psi.t())*DHDk1*psi);
    arma::cx_mat v2 = ((psi.t())*DHDk2*psi);
    return arma::vec3{real(v0(0,0)), real(v1(0,0)), real(v2(0,0))};
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::outputBandStructureAlongHighSymmetry(const char* inputFilename, const char* outputFilename, int numStepsFirstLine)
{
    std::ifstream file(inputFilename); // Replace with the actual file name
    std::string line;
    bool capture = false;
    // The following structure contains k123 coordinates and their corresponding Cartesian ones.
    struct KPointPair {
        std::string capturedInformation;
        arma::vec3 k_start;
        arma::vec3 k_end;
        arma::vec3 start_cartesian;
        arma::vec3 end_cartesian;
        int steps;
    };
    std::vector<KPointPair> kpointPairs;
    arma::mat B(3,3);
    B.col(0) = this->b1;
    B.col(1) = this->b2;
    B.col(2) = this->b3;

    //B.print("debug B = ");
    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }

    while (getline(file, line)) {
        if (line == "begin kpoint_path") {
            capture = true;
            continue; // Skip the begin line
        }
        if (line == "end kpoint_path") {
            break; // Stop reading when end line is encountered
        }
        if (capture) {
            KPointPair pair;
            pair.capturedInformation = line;
            std::istringstream iss(line);
            arma::vec start(3), end(3);
            char dummy; // To ignore the letters
            iss >> dummy >> pair.k_start[0] >> pair.k_start[1] >> pair.k_start[2]
                >> dummy >> pair.k_end[0] >> pair.k_end[1] >> pair.k_end[2];

            // Transform to Cartesian coordinates for distance calculation
            pair.start_cartesian = B * pair.k_start;
            pair.end_cartesian = B * pair.k_end;
            //pair.start_cartesian = b1 * pair.k_start[0] + b2 * pair.k_start[1] + b3 * pair.k_start[2];
            //pair.end_cartesian = b1 * pair.k_end[0] + b2 * pair.k_end[1] + b3 * pair.k_end[2];

            kpointPairs.push_back(pair);
        }
    }
    file.close();
    double firstLineDistance = arma::norm(kpointPairs[0].start_cartesian - kpointPairs[0].end_cartesian);
    // Calculate and output the number of steps for each line
    for (auto& pair : kpointPairs) {
        double distance = arma::norm(pair.start_cartesian - pair.end_cartesian);
        pair.steps = static_cast<int>(std::ceil(numStepsFirstLine * distance / firstLineDistance));
        //std::cout << "Kxyz-Start: " << pair.start_cartesian.t() << " Kxyz-End: " << pair.end_cartesian.t() << " Steps: " << pair.steps << std::endl;
    }

    // Output the captured paths and their steps:
    int cumulativeSteps = 0;

    std::cout << "======== Writing band structure to " << outputFilename << " ========\n";
    for (const auto& pair : kpointPairs) {
        cumulativeSteps += pair.steps; // Accumulate the steps
        std::cout << pair.capturedInformation << "; steps end at: " << cumulativeSteps << std::endl;
    }
    // Now we get ready to write the band structure to the output file.
    int kPointCount = 0;
    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open output file" << std::endl;
        return;
    }
    outputFile << std::scientific; // Set the output format to scientific.

    for (const auto& pair : kpointPairs) {
        arma::vec3 delta = (pair.k_end - pair.k_start) / pair.steps;

        // Iterate through each step
        for (int step = 0; step < pair.steps; step++) {
            arma::vec3 current_k = pair.k_start + step * delta;
            arma::cx_mat H = this->generateLocalHamiltonian(current_k(0), current_k(1), current_k(2));
            arma::vec energy;
            arma::cx_mat eigenvectors;
            //std::cout << "debug calculating " << step << "\n";
            eig_sym(energy, eigenvectors, H);

            // Write k-point count and eigenvalues to output file
            outputFile << ++kPointCount;
            for (double eigenvalue : energy) {
                outputFile << " " << eigenvalue;
            }
            outputFile << std::endl;
        }
    }
    outputFile.close();
    std::cout << "======== Band structure written to " << outputFilename << " ========\n\n" << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/*void Bulk::initializeTheMesh(unsigned long num_of_points) {
    this->theMesh = initializeMesh(*this, num_of_points);
}*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::initializeTheMeshParallel(unsigned long num_of_points, int num_OpenMP_threads) {
    this->theMesh = initializeMeshParallel(*this, num_of_points, num_OpenMP_threads);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::refineTheMeshForOneBandParallel(int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads) {
    std::cout << "======== Refine band " << iBand << " ========" << std::endl;
    refineMeshParallel(this->theMesh, *this, iBand, num_refine, Ef, num_OpenMP_threads);
    std::cout << "======== Refine band " << iBand << " finished ========\n\n" << std:: endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::refineMeshParallelForBerryPhase(int num_refine, arma::uword iBand, double Ef, int num_OpenMP_threads, int depth) {
    std::cout << "======== Refine band " << iBand << " for Berry curvature. ========" << std::endl;
    refineMeshParallelForBerryPhaseImplementation(this->theMesh, *this, iBand, num_refine, Ef, num_OpenMP_threads, depth);
    std::cout << "======== Refine band " << iBand << " finished ========\n\n" << std:: endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::writeMeshToVtu(const std::string filename, arma::uword iBand)  {
    Iso_cuboid domain(P3_Point(-0.5, -0.5, -0.5), P3_Point(0.5, 0.5, 0.5));
    std::cout << "Writing mesh to file " << filename << "..." << std::flush;
    write_to_vtu_P3(filename, this->theMesh, domain, iBand, true, this->B);
    std::cout << "done." << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::findExtrema(arma::uword iBand)  {
    double e_min = std::numeric_limits<double>::infinity();
    double e_max = -std::numeric_limits<double>::infinity();
    for (auto it = this->theMesh.unique_vertices_begin(); it != this->theMesh.unique_vertices_end(); ++it )  {
        arma::vec eig = it->info().eigVal;
        double value = eig(iBand);
        if (value < e_min) e_min = value;
        if (value > e_max) e_max = value;
    }
    std::cout << "Minimum value for band " << iBand << ": " << e_min << std::endl;
    std::cout << "Maximum value for band " << iBand << ": " << e_max << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
arma::vec3 Bulk::get_b1(void)  {
    return this->b1;
}

arma::vec3 Bulk::get_b2(void)  {
    return this->b2;
}

arma::vec3 Bulk::get_b3(void)   {
    return this->b3;
}

arma::mat33 Bulk::get_B(void)   {
    return this->B;
}

arma::cx_mat Bulk::get_H(void)    {
    return this->Hamiltonian;
}

const arma::uword Bulk::get_matrixN(void)    {
    return this->matrixN;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::calculateBerryFluxBZ() {
    std::cout << "======== Calculating Berry flux for all facets and vertices ========" << std::endl;
    if (this->theMesh.number_of_vertices() == 0) {
        return;
    }
    if (!this->theMesh.is_1_cover())
        this->theMesh.convert_to_1_sheeted_covering();

    // The all-band Berry flux needs full eigenvectors only for this call.
    std::unordered_map<P3_Delaunay::Vertex_handle, arma::cx_mat, CGAL::Handle_hash_function> states;
    states.reserve(this->theMesh.number_of_vertices());
    for (auto vertex = this->theMesh.unique_vertices_begin(); vertex != this->theMesh.unique_vertices_end(); ++vertex) {
        const auto& k = vertex->point();
        const auto hamiltonian = this->generateLocalHamiltonian(k.x(), k.y(), k.z());
        arma::vec energies;
        auto& eigenvectors = states[vertex];
        if (!hamiltonian.is_finite() || !arma::eig_sym(energies, eigenvectors, hamiltonian)) {
            throw std::runtime_error("Could not calculate eigenstates for bulk Berry flux.");
        }
    }

    std::cout << "Refreshing all vertices and cells..." << std::flush;
    for (auto cit = this->theMesh.unique_cells_begin(); cit != this->theMesh.unique_cells_end(); ++cit) {
        cit->info().facetBerryFlux.resize(4);
        for (int i=0; i<4; i++)
            cit->info().facetBerryFlux[i].set_size(this->matrixN);
    }
    std::cout << " done." << std::endl;
    unsigned long long count = 0;
    for (auto vit = this->theMesh.unique_vertices_begin(); vit!=this->theMesh.unique_vertices_end(); ++vit) {
        std::vector<P3_Delaunay::Cell_handle> cells;
        this->theMesh.incident_cells(vit, std::back_inserter(cells));
        vit->info().berryFlux.set_size(this->matrixN);
        vit->info().berryFlux.zeros();
        for (const auto &ch: cells) {
            int index = ch->index(vit);
            ch->info().facetBerryFlux[index].resize(this->matrixN);
            int ind0 = CGAL::Triangulation_utils_3::vertex_triple_index(index, 2);
            int ind1 = CGAL::Triangulation_utils_3::vertex_triple_index(index, 1);
            int ind2 = CGAL::Triangulation_utils_3::vertex_triple_index(index, 0);
            // Note tat we want the right-hand rule with the thumb being the normal vector pointing outwards.
            // This is opposite to CGAL's ccw implementation.
            const auto& states0 = states.at(this->theMesh.get_original_vertex(ch->vertex(ind0)));
            const auto& states1 = states.at(this->theMesh.get_original_vertex(ch->vertex(ind1)));
            const auto& states2 = states.at(this->theMesh.get_original_vertex(ch->vertex(ind2)));
            for (arma::uword i = 0; i < this->matrixN; i++) {
                const auto psi0 = states0.col(i);
                const auto psi1 = states1.col(i);
                const auto psi2 = states2.col(i);
                const double flux = -std::imag(arma::cdot(psi0, psi1) * arma::cdot(psi1, psi2) * arma::cdot(psi2, psi0));
                vit->info().berryFlux(i) += flux;
                ch->info().facetBerryFlux[index](i) = flux;
            }
        }
        count++;
        if (count % 10000 == 0)
            std::cout << "debug " << count << std::endl;
    }
    std::cout << "done." << std::endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

double occupiedTetrahedronFraction(const std::array<double, 4>& e, double ef) {
    // The energies are sorted. A flat band contributes only below the Fermi level.
    if (ef <= e[0]) {
        return 0.0;
    }
    if (ef >= e[3]) {
        return 1.0;
    }
    if (ef <= e[1]) {
        return (ef - e[0]) / (e[1] - e[0]) * (ef - e[0]) / (e[2] - e[0]) *
               (ef - e[0]) / (e[3] - e[0]);
    }
    if (ef >= e[2]) {
        return 1.0 - (e[3] - ef) / (e[3] - e[0]) * (e[3] - ef) / (e[3] - e[1]) *
                     (e[3] - ef) / (e[3] - e[2]);
    }

    // Two occupied vertices: the clipped prism decomposes into three tetrahedra.
    const double t02 = (ef - e[0]) / (e[2] - e[0]);
    const double t03 = (ef - e[0]) / (e[3] - e[0]);
    const double t12 = (ef - e[1]) / (e[2] - e[1]);
    const double t13 = (ef - e[1]) / (e[3] - e[1]);
    return std::min(1.0, t02 * t03 + t02 * t13 * (1.0 - t03) + t12 * t13 * (1.0 - t02));
}

} // namespace

std::vector<arma::mat33> Bulk::intrinsicMatrixOfFermiSeaIntegral(
    const std::vector<double>& fermiEnergies, int numThreads) {
    if (numThreads < 1 || !std::all_of(fermiEnergies.begin(), fermiEnergies.end(),
                                      [](double ef) { return std::isfinite(ef); })) {
        throw std::invalid_argument("Fermi-sea integration requires finite energies and a positive thread count.");
    }
    if (fermiEnergies.empty()) {
        return {};
    }
    if (theMesh.number_of_cells() == 0) {
        throw std::runtime_error("Initialize the bulk mesh before evaluating the Fermi-sea integral.");
    }
    for (auto v = theMesh.unique_vertices_begin(); v != theMesh.unique_vertices_end(); ++v) {
        if (v->info().eigVal.n_elem != matrixN || !v->info().eigVal.is_finite()) {
            throw std::runtime_error("Fermi-sea integration requires valid band energies on the bulk mesh.");
        }
    }

    std::vector<std::size_t> energyOrder(fermiEnergies.size());
    std::iota(energyOrder.begin(), energyOrder.end(), 0);
    std::sort(energyOrder.begin(), energyOrder.end(), [&](std::size_t a, std::size_t b) {
        return fermiEnergies[a] < fermiEnergies[b];
    });
    std::vector<double> sortedEnergies;
    sortedEnergies.reserve(energyOrder.size());
    for (const auto index : energyOrder) {
        sortedEnergies.push_back(fermiEnergies[index]);
    }

    std::vector<P3_Delaunay::Cell_handle> cells;
    cells.reserve(theMesh.number_of_cells());
    for (auto cell = theMesh.unique_cells_begin(); cell != theMesh.unique_cells_end(); ++cell) {
        cells.push_back(cell);
    }

    const arma::vec3 zero(arma::fill::zeros);
    std::vector<arma::vec3> integral(sortedEnergies.size(), zero);
    double totalVolume = 0.0;
    std::exception_ptr failure;
    #pragma omp parallel num_threads(numThreads) reduction(+:totalVolume)
    {
        std::vector<arma::vec3> localIntegral(sortedEnergies.size(), zero);
        arma::mat fractions(sortedEnergies.size(), matrixN);
        #pragma omp for
        for (std::size_t cellIndex = 0; cellIndex < cells.size(); ++cellIndex) {
            try {
                const auto cell = cells[cellIndex];
                std::array<arma::vec3, 4> points;
                std::array<const arma::vec*, 4> vertexEnergies;
                for (int i = 0; i < 4; ++i) {
                    const auto p = theMesh.point(theMesh.periodic_point(cell, i));
                    points[i] = {p.x(), p.y(), p.z()};
                    vertexEnergies[i] = &theMesh.get_original_vertex(cell->vertex(i))->info().eigVal;
                }
                const double volume = std::abs(arma::dot(points[1] - points[0],
                    arma::cross(points[2] - points[0], points[3] - points[0]))) / 6.0;
                totalVolume += volume;

                // Linear band energies give exact occupation fractions for the interpolant.
                for (arma::uword band = 0; band < matrixN; ++band) {
                    std::array<double, 4> e{(*vertexEnergies[0])(band), (*vertexEnergies[1])(band),
                                            (*vertexEnergies[2])(band), (*vertexEnergies[3])(band)};
                    std::sort(e.begin(), e.end());
                    for (std::size_t i = 0; i < sortedEnergies.size(); ++i) {
                        fractions(i, band) = occupiedTetrahedronFraction(e, sortedEnergies[i]);
                    }
                }
                if (fractions(sortedEnergies.size() - 1, 0) == 0.0 || fractions(0, matrixN - 1) == 1.0) {
                    continue;
                }

                // Use the same centroid curvature for all energies, with lifted periodic geometry.
                const arma::vec3 k = (points[0] + points[1] + points[2] + points[3]) / 4.0;
                const auto h = generateLocalHamiltonian(k(0), k(1), k(2));
                arma::vec energies;
                arma::cx_mat frame;
                if (!h.is_finite() || !arma::eig_sym(energies, frame, h) || !energies.is_finite()) {
                    throw std::runtime_error("Could not calculate the Fermi-sea integrand at a tetrahedron centroid.");
                }
                auto derivatives = generateLocalDHDk(k(0), k(1), k(2));
                for (auto& derivative : derivatives) {
                    derivative = frame.t() * derivative * frame;
                }

                // Use one occupation and energy per numerically degenerate eigenspace so
                // the interspace traces are invariant under rotations of its eigenvectors.
                const double gapTolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                                            std::max(std::abs(energies.front()), std::abs(energies.back()));
                for (arma::uword first = 0; first < matrixN;) {
                    arma::uword end = first + 1;
                    while (end < matrixN && energies(end) - energies(end - 1) <= gapTolerance) {
                        ++end;
                    }
                    if (end - first > 1) {
                        auto group = fractions.cols(first, end - 1);
                        const arma::vec weight = arma::mean(group, 1);
                        group.each_col() = weight;
                        auto groupEnergies = energies.subvec(first, end - 1);
                        const double representativeEnergy = arma::mean(groupEnergies);
                        groupEnergies.fill(representativeEnergy);
                    }
                    first = end;
                }

                const auto lastEnergy = sortedEnergies.size() - 1;
                for (arma::uword n = 0; n < matrixN && fractions(lastEnergy, n) > 0.0; ++n) {
                    for (arma::uword m = n + 1; m < matrixN; ++m) {
                        const double gap = energies(m) - energies(n);
                        if (gap <= gapTolerance || fractions(0, m) == 1.0) {
                            continue;
                        }
                        // Opposite contributions from n and m cancel where both have the same occupation.
                        const double* lower = fractions.colptr(n);
                        const double* upper = fractions.colptr(m);
                        const auto begin = std::upper_bound(lower, lower + sortedEnergies.size(), 0.0) - lower;
                        const auto end = std::lower_bound(upper, upper + sortedEnergies.size(), 1.0) - upper;
                        if (begin >= end) {
                            continue;
                        }
                        const auto x = derivatives[0](n, m) / gap;
                        const auto y = derivatives[1](n, m) / gap;
                        const auto z = derivatives[2](n, m) / gap;
                        const arma::vec3 flux = -2.0 * volume * arma::vec3{
                            std::imag(y * std::conj(z)), std::imag(z * std::conj(x)), std::imag(x * std::conj(y))};
                        if (!flux.is_finite()) {
                            throw std::runtime_error("Nonfinite Berry curvature in the Fermi-sea integral.");
                        }
                        for (auto i = begin; i < end; ++i) {
                            localIntegral[i] += (lower[i] - upper[i]) * flux;
                        }
                    }
                }
            } catch (...) {
                #pragma omp critical(fermiSeaFailure)
                {
                    if (!failure) {
                        failure = std::current_exception();
                    }
                }
            }
        }
        #pragma omp critical(fermiSeaReduction)
        for (std::size_t i = 0; i < integral.size(); ++i) {
            integral[i] += localIntegral[i];
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    const double domainVolume = theMesh.domain().volume();
    if (!std::isfinite(totalVolume) || std::abs(totalVolume - domainVolume) > 1e-8 * domainVolume) {
        throw std::runtime_error("The unique periodic tetrahedra do not cover one Brillouin zone.");
    }

    // The curvature pseudovector transforms by B / det(B), while d^3k scales by |det(B)|.
    const double handedness = std::copysign(1.0, arma::det(B));
    std::vector<arma::mat33> result(fermiEnergies.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const arma::vec3 flux = handedness * B * integral[i];
        result[energyOrder[i]] = {{0.0, flux(2), -flux(1)}, {-flux(2), 0.0, flux(0)},
                                  {flux(1), -flux(0), 0.0}};
    }
    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
void Bulk::setSmallSpinSplitting(double eta) {
    this->spinSplittingEta = eta;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
