#include "TriangleWilsonLoop.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Frames = std::array<arma::cx_mat, 3>;
using Energies = std::array<arma::vec, 3>;
constexpr double phaseTolerance = 1e-12;
const double pi = std::acos(-1.0);

void expect_near(double actual, double expected, const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > phaseTolerance) {
        throw std::runtime_error(message + ": expected " + std::to_string(expected)
                                 + ", got " + std::to_string(actual));
    }
}

quantumgeom::TriangleWilsonPhases phases(const Frames& frames, const Energies& energies,
                                        std::array<std::size_t, 3> order = {0, 1, 2}) {
    return quantumgeom::triangleWilsonPhases(
        {&frames[order[0]], &frames[order[1]], &frames[order[2]]},
        {&energies[order[0]], &energies[order[1]], &energies[order[2]]}, 1e-8);
}

Frames independent_loops() {
    Frames frames;
    for (auto& frame : frames) {
        frame.zeros(4, 2);
    }
    frames[0](0, 0) = 1.0;
    frames[0](2, 1) = 1.0;
    frames[1](0, 0) = frames[1](1, 0) = 1.0 / std::sqrt(2.0);
    frames[1](2, 1) = frames[1](3, 1) = 1.0 / std::sqrt(2.0);
    frames[2](0, 0) = 1.0 / std::sqrt(2.0);
    frames[2](1, 0) = arma::cx_double(0.0, 1.0 / std::sqrt(2.0));
    const double norm = std::hypot(1.0, 0.25);
    frames[2](2, 1) = 1.0 / norm;
    frames[2](3, 1) = arma::cx_double(0.0, -0.25 / norm);
    return frames;
}

arma::cx_mat rotate_pair(double angle, double phase) {
    const auto offDiagonal = std::polar(std::sin(angle), phase);
    arma::cx_mat rotation(2, 2);
    rotation(0, 0) = rotation(1, 1) = std::cos(angle);
    rotation(1, 0) = offDiagonal;
    rotation(0, 1) = -std::conj(offDiagonal);
    return rotation;
}

void test_rank_one_phase() {
    const auto pair = independent_loops();
    const Frames frames{pair[0].col(0), pair[1].col(0), pair[2].col(0)};
    const Energies energies{arma::vec{0.0}, arma::vec{0.0}, arma::vec{0.0}};
    const auto result = phases(frames, energies);
    expect_near(result.total, -pi / 4.0, "Rank-one phase follows the Fermi-surface sign convention");
    expect_near(result.shares(0), result.total, "Rank-one share equals the total");
    expect_near(result.minSingularValue, 1.0 / std::sqrt(2.0), "Rank-one minimum overlap");
}

void test_independent_loops() {
    const auto frames = independent_loops();
    const arma::vec levels{-1.0, 1.0};
    const Energies energies{levels, levels, levels};
    const auto result = phases(frames, energies);
    expect_near(result.shares(0), -pi / 4.0, "First independent band's phase");
    expect_near(result.shares(1), std::atan(0.25), "Second independent band's phase");
    expect_near(result.total, arma::accu(result.shares), "Unwrapped shares sum to their multiplet phase");
    if (result.nearBranchCut) {
        throw std::runtime_error("Small Wilson eigenphases should be far from the branch cut");
    }
}

void test_mixed_energy_frame() {
    auto frames = independent_loops();
    frames[1] *= rotate_pair(pi / 4.0, 0.3);
    const arma::vec levels{-1.0, 1.0};
    const Energies energies{levels, levels, levels};
    const auto result = phases(frames, energies);
    const auto cyclic = phases(frames, energies, {1, 2, 0});
    const auto reversed = phases(frames, energies, {0, 2, 1});

    // Vertices 0 and 2 match the Wilson eigenvectors, while vertex 1 splits
    // both eigenphases evenly between its two local energy eigenvectors.
    const double first = -pi / 4.0;
    const double second = std::atan(0.25);
    expect_near(result.shares(0), (5.0 * first + second) / 6.0, "First overlap-weighted share");
    expect_near(result.shares(1), (first + 5.0 * second) / 6.0, "Second overlap-weighted share");
    expect_near(result.total, first + second, "Mixing the energy basis preserves the Wilson spectrum");
    expect_near(cyclic.total, result.total, "Multiplet phase is invariant under a cyclic permutation");
    expect_near(reversed.total, -result.total, "Reversing orientation negates the multiplet phase");
    for (arma::uword band = 0; band < result.shares.n_elem; ++band) {
        expect_near(cyclic.shares(band), result.shares(band), "Cyclic permutation preserves the shares");
        expect_near(reversed.shares(band), -result.shares(band), "Orientation reverses each share");
    }
}

void test_degenerate_block() {
    auto frames = independent_loops();
    const arma::vec levels{0.0, 0.0};
    const Energies energies{levels, levels, levels};
    const auto unrotated = phases(frames, energies);
    frames[0] *= rotate_pair(0.7, 0.4);
    frames[1] *= rotate_pair(1.2, 0.2);
    frames[2] *= rotate_pair(0.3, 2.0);
    const auto rotated = phases(frames, energies);
    expect_near(rotated.total, unrotated.total, "Arbitrary degenerate-frame rotations preserve total flux");
    for (arma::uword band = 0; band < rotated.shares.n_elem; ++band) {
        expect_near(rotated.shares(band), rotated.total / 2.0, "Degenerate block receives equal shares");
        expect_near(rotated.shares(band), unrotated.shares(band), "Degenerate shares are gauge invariant");
    }
}

void test_full_space() {
    const Frames frames{rotate_pair(0.7, 0.4), rotate_pair(1.2, 0.2), rotate_pair(0.3, 2.0)};
    const arma::vec levels{-1.0, 1.0};
    const Energies energies{levels, levels, levels};
    const auto result = phases(frames, energies);
    expect_near(result.total, 0.0, "A loop transporting the entire fixed Hilbert space is trivial");
    expect_near(arma::norm(result.shares), 0.0, "The full-space loop contains no individual-band flux");
    expect_near(result.minSingularValue, 1.0, "A full-space overlap is already unitary");
}

void test_branch_cut_and_singular_link() {
    Frames frames;
    const Energies energies{arma::vec{0.0}, arma::vec{0.0}, arma::vec{0.0}};
    for (std::size_t vertex = 0; vertex < frames.size(); ++vertex) {
        const double angle = 2.0 * pi * static_cast<double>(vertex) / 3.0;
        frames[vertex].set_size(2, 1);
        frames[vertex](0, 0) = std::cos(angle);
        frames[vertex](1, 0) = std::sin(angle);
    }
    const auto result = phases(frames, energies);
    expect_near(std::abs(result.total), pi, "Three negative edge overlaps give phase pi modulo 2pi");
    if (!result.nearBranchCut) {
        throw std::runtime_error("A Wilson eigenphase at pi must be flagged");
    }

    frames[1](0, 0) = 0.0;
    frames[1](1, 0) = 1.0;
    try {
        phases(frames, energies);
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("An exactly singular overlap must be rejected");
}

void test_determinant_branch() {
    Frames frames;
    const arma::vec levels{-1.0, 1.0};
    const Energies energies{levels, levels, levels};
    for (std::size_t vertex = 0; vertex < frames.size(); ++vertex) {
        auto& frame = frames[vertex];
        frame.zeros(4, 2);
        const double angle = 2.0 * pi * static_cast<double>(vertex) / 3.0;
        frame(0, 0) = frame(2, 1) = std::sqrt(0.6);
        frame(1, 0) = frame(3, 1) = std::polar(std::sqrt(0.4), angle);
    }
    const auto result = phases(frames, energies);
    const double eigenphase = -3.0 * std::arg(0.6 + std::polar(0.4, 2.0 * pi / 3.0));
    expect_near(result.shares(0), eigenphase, "First share uses its Wilson eigenphase branch");
    expect_near(result.shares(1), eigenphase, "Second share uses its Wilson eigenphase branch");
    expect_near(result.total, 2.0 * eigenphase + 2.0 * pi, "The determinant uses its principal branch");
    if (!result.nearBranchCut) {
        throw std::runtime_error("A sum of Wilson eigenphases outside the determinant branch must be flagged");
    }
}

void test_mismatched_frames() {
    auto frames = independent_loops();
    frames[1].shed_col(1);
    const Energies energies{arma::vec{-1.0, 1.0}, arma::vec{-1.0}, arma::vec{-1.0, 1.0}};
    try {
        phases(frames, energies);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Unequal subspace ranks at triangle vertices must be rejected");
}

}  // namespace

int main() {
    try {
        test_rank_one_phase();
        test_independent_loops();
        test_mixed_energy_frame();
        test_degenerate_block();
        test_full_space();
        test_branch_cut_and_singular_link();
        test_determinant_branch();
        test_mismatched_frames();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
