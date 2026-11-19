#include "FermiSurfaceExtraction.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected a basis name");
        const std::string name = argv[1];
        arma::mat33 basis(arma::fill::eye);
        double expected_area = 3.0;
        if (name == "scaled") {
            basis *= 2.0;
            expected_area = 12.0;
        } else if (name == "sheared") {
            basis = arma::mat33{{1, 1, 0}, {0, 2, 0}, {0, 0, 1}};
            expected_area = 6.0;
        } else if (name != "identity") {
            throw std::runtime_error("Unknown basis: " + name);
        }

        // The z=1 face has perpendicular edges of lengths 2 and 3. Its
        // circumsphere is clear of periodic copies in this larger domain.
        P3_Delaunay mesh(Iso_cuboid(0, 0, 0, 10, 10, 10));
        mesh.insert(P3_Point(1, 1, 1));
        mesh.insert(P3_Point(3, 1, 1));
        mesh.insert(P3_Point(1, 4, 1));
        mesh.insert(P3_Point(1, 1, 5));
        auto cell = mesh.locate(P3_Point(1.5, 1.75, 2));

        // Identify the face from actual periodic coordinates instead of
        // assuming CGAL's vertex ordering or sheet representation.
        int opposite_vertex = -1;
        int base_vertices = 0;
        BulkBandEigenstates states;
        for (int j = 0; j < 4; ++j) {
            const auto point = mesh.point(mesh.periodic_point(cell, j));
            if (point == P3_Point(1, 1, 5)) {
                opposite_vertex = j;
            } else if (point == P3_Point(1, 1, 1) ||
                       point == P3_Point(3, 1, 1) ||
                       point == P3_Point(1, 4, 1)) {
                ++base_vertices;
            } else {
                throw std::runtime_error("Located an unexpected periodic tetrahedron");
            }
            auto& state = states[mesh.get_original_vertex(cell->vertex(j))];
            state.zeros(2);
            state(0) = 1.0;
        }
        if (opposite_vertex < 0 || base_vertices != 3) {
            throw std::runtime_error("Missing the known-area triangle");
        }

        const int ind1 = CGAL::Triangulation_utils_3::vertex_triple_index(opposite_vertex, 1);
        const int ind2 = CGAL::Triangulation_utils_3::vertex_triple_index(opposite_vertex, 0);
        const double s = 1.0 / std::sqrt(2.0);
        auto& psi1 = states.at(mesh.get_original_vertex(cell->vertex(ind1)));
        auto& psi2 = states.at(mesh.get_original_vertex(cell->vertex(ind2)));
        psi1(0) = s;
        psi1(1) = s;
        psi2(0) = s;
        psi2(1) = std::complex<double>(0, s);

        // These normalized states give a loop overlap (1+i)/4. Keep that
        // numerator fixed so this regression isolates geometric normalization.
        arma::vec4 curvature;
        berryCurvatureSingleCell(mesh, cell, states, curvature, basis);
        const double expected = -0.25 / expected_area;
        const double actual = curvature(opposite_vertex);
        std::cout << name << ": expected area=" << expected_area
                  << ", expected curvature=" << expected << ", actual=" << actual << '\n';
        if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-13) {
            throw std::runtime_error("Incorrect triangle-area normalization");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
