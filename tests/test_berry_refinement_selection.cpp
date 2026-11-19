#include "FermiSurfaceExtraction.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Cell = P3_Delaunay::Cell_handle;

std::vector<Cell> make_cells(P3_Delaunay& mesh, int count, BulkBandEigenstates& states) {
    // These tetrahedra have disjoint empty circumspheres, so all survive
    // insertion. Locate only after insertion has finished to keep handles valid.
    for (int i = 0; i < count; ++i) {
        const double x = 1 + 10 * i;
        mesh.insert(P3_Point(x, 1, 1));
        mesh.insert(P3_Point(x + 2, 1, 1));
        mesh.insert(P3_Point(x, 4, 1));
        mesh.insert(P3_Point(x, 1, 5));
    }

    std::vector<Cell> cells;
    for (int i = 0; i < count; ++i) {
        const double x = 1 + 10 * i;
        auto cell = mesh.locate(P3_Point(x + 0.5, 1.75, 2));
        int opposite = -1;
        int base_vertices = 0;
        for (int j = 0; j < 4; ++j) {
            const auto point = mesh.point(mesh.periodic_point(cell, j));
            if (point == P3_Point(x, 1, 5)) {
                opposite = j;
            } else if (point == P3_Point(x, 1, 1) ||
                       point == P3_Point(x + 2, 1, 1) ||
                       point == P3_Point(x, 4, 1)) {
                ++base_vertices;
            } else {
                throw std::runtime_error("Unexpected periodic tetrahedron");
            }
            auto& state = states[mesh.get_original_vertex(cell->vertex(j))];
            state.zeros(2);
            state(0) = 1.0;
        }
        if (opposite < 0 || base_vertices != 3) {
            throw std::runtime_error("Incomplete tetrahedron fixture");
        }

        const int ind1 = CGAL::Triangulation_utils_3::vertex_triple_index(opposite, 1);
        const int ind2 = CGAL::Triangulation_utils_3::vertex_triple_index(opposite, 0);
        const double s = 1.0 / std::sqrt(2.0);
        auto& psi1 = states.at(mesh.get_original_vertex(cell->vertex(ind1)));
        auto& psi2 = states.at(mesh.get_original_vertex(cell->vertex(ind2)));
        psi1(0) = s;
        psi1(1) = s;
        psi2(0) = s;
        psi2(1) = s * std::exp(std::complex<double>(0, 0.4 * i));
        cells.push_back(cell);
    }
    return cells;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Expected count or ranking");
        const std::string mode = argv[1];
        P3_Delaunay mesh(Iso_cuboid(0, 0, 0, 50, 50, 50));
        arma::mat33 basis(arma::fill::eye);
        BulkBandEigenstates states;
        if (mode == "count" && argc == 5) {
            const size_t total = std::stoull(argv[2]);
            const int depth = std::stoi(argv[3]);
            const size_t expected = std::stoull(argv[4]);
            const auto cells = make_cells(mesh, 1, states);
            // Repeated valid handles isolate selection counts from ranking.
            std::vector<Cell> candidates(total, cells.front());
            const auto selected = topListOfLargeBerryCurvatures(candidates, mesh, states, basis, depth);
            std::cout << "candidates=" << total << ", depth=" << depth
                      << ", expected=" << expected << ", selected=" << selected.size() << '\n';
            if (selected.size() != expected ||
                !std::all_of(selected.begin(), selected.end(),
                             [&](Cell cell) { return cell == cells.front(); })) {
                throw std::runtime_error("Incorrect refinement selection count or handle");
            }
        } else if (mode == "ranking" && argc == 2) {
            auto cells = make_cells(mesh, 4, states);
            double previous_score = -1;
            for (auto cell : cells) {
                arma::vec4 curvature;
                berryCurvatureSingleCell(mesh, cell, states, curvature, basis);
                const double score = arma::max(arma::abs(curvature));
                if (!std::isfinite(score) || score <= previous_score) {
                    throw std::runtime_error("Fixture must have distinct increasing scores");
                }
                previous_score = score;
            }
            // Increasing complex phases give increasing curvature magnitudes.
            // Shuffle candidates and check identities rather than output order.
            std::vector<Cell> candidates{cells[2], cells[0], cells[3], cells[1]};
            const auto selected = topListOfLargeBerryCurvatures(candidates, mesh, states, basis, 50);
            if (selected.size() != 2 ||
                std::count(selected.begin(), selected.end(), cells[2]) != 1 ||
                std::count(selected.begin(), selected.end(), cells[3]) != 1) {
                throw std::runtime_error("Did not select the two highest-curvature cells exactly once");
            }
            std::cout << "Selected the two highest-curvature cells exactly once\n";
        } else {
            throw std::runtime_error("Invalid test arguments");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
