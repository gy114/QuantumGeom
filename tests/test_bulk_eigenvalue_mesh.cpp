#include "Bulk.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check_eigenvalues(Bulk& bulk) {
    const auto& mesh = bulk.theMesh;
    for (auto vertex = mesh.unique_vertices_begin(); vertex != mesh.unique_vertices_end(); ++vertex) {
        const auto& eigenvalues = vertex->info().eigVal;
        if (eigenvalues.n_elem != 2 || !eigenvalues.is_finite()) {
            throw std::runtime_error("Every mesh vertex should retain two finite eigenvalues");
        }

        // Compute the spectrum of this two-orbital Hamiltonian independently
        // of the eigensolver used to populate the mesh.
        const auto& point = vertex->point();
        const auto h = bulk.generateLocalHamiltonian(point.x(), point.y(), point.z());
        const double mean = (h(0, 0).real() + h(1, 1).real()) / 2;
        const double half_gap = std::hypot((h(0, 0).real() - h(1, 1).real()) / 2, std::abs(h(0, 1)));
        if (std::abs(eigenvalues(0) - (mean - half_gap)) > 1e-11 ||
            std::abs(eigenvalues(1) - (mean + half_gap)) > 1e-11) {
            throw std::runtime_error("Stored spectrum does not match the Hamiltonian at its vertex");
        }
    }
}

void check_bulk_berry_flux(Bulk& bulk) {
    bulk.calculateBerryFluxBZ();
    const auto& mesh = bulk.theMesh;
    bool nonzero_vertex_flux = false;
    for (auto vertex = mesh.unique_vertices_begin(); vertex != mesh.unique_vertices_end(); ++vertex) {
        const auto& flux = vertex->info().berryFlux;
        if (flux.n_elem != 2 || !flux.is_finite()) {
            throw std::runtime_error("Bulk Berry flux should be finite for both bands at every vertex");
        }
        nonzero_vertex_flux |= arma::norm(flux, "inf") > 1e-10;
    }

    bool nonzero_facet_flux = false;
    for (auto cell = mesh.unique_cells_begin(); cell != mesh.unique_cells_end(); ++cell) {
        const auto& facets = cell->info().facetBerryFlux;
        if (facets.size() != 4) {
            throw std::runtime_error("Each tetrahedron should have flux for four facets");
        }
        for (const auto& flux : facets) {
            if (flux.n_elem != 2 || !flux.is_finite()) {
                throw std::runtime_error("Bulk Berry flux should be finite for both bands at every facet");
            }
            nonzero_facet_flux |= arma::norm(flux, "inf") > 1e-10;
        }
    }
    if (!nonzero_vertex_flux || !nonzero_facet_flux) {
        throw std::runtime_error("Expected nontrivial vertex and facet Berry flux in the Weyl model");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Expected a path to the two-band Weyl model");
        }
        Bulk bulk(std::string{argv[1]});
        constexpr arma::uword band = 1;
        constexpr double fermi_energy = 3.0;
        bulk.initializeTheMeshParallel(250, 2);
        const auto initial_size = bulk.theMesh.number_of_vertices();
        if (initial_size == 0 || bulk.get_matrixN() != 2) {
            throw std::runtime_error("Expected a nonempty mesh for the two-band model");
        }
        check_eigenvalues(bulk);

        bulk.refineTheMeshForOneBandParallel(1, band, fermi_energy, 2);
        const auto refined_size = bulk.theMesh.number_of_vertices();
        if (refined_size <= initial_size) {
            throw std::runtime_error("Energy refinement should add vertices");
        }
        check_eigenvalues(bulk);

        // An intermediate percentage requires actual Berry-curvature ranking.
        bulk.refineMeshParallelForBerryPhase(2, band, fermi_energy, 2, 20);
        if (bulk.theMesh.number_of_vertices() <= refined_size) {
            throw std::runtime_error("Berry-guided refinement should add vertices to the energy mesh");
        }
        check_eigenvalues(bulk);
        check_bulk_berry_flux(bulk);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
