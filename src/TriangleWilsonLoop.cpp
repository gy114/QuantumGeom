/*
 * Project Name: QuantumGeom
 * Copyright (C) 2026 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "TriangleWilsonLoop.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace quantumgeom {
namespace {

const double pi = std::acos(-1.0);
constexpr double minOverlap = 1e-10;

void checkOverlap(double smallestSingularValue) {
    if (!std::isfinite(smallestSingularValue) || smallestSingularValue <= minOverlap) {
        throw std::runtime_error("Rank-deficient Berry overlap. Refine the mesh or review the band group.");
    }
}

arma::cx_mat unitaryOverlap(const arma::cx_mat& from, const arma::cx_mat& to,
                           double& minSingularValue) {
    const arma::cx_mat overlap = from.t() * to;
    arma::cx_mat left, right;
    arma::vec singularValues;
    if (!arma::svd(left, singularValues, right, overlap)) {
        throw std::runtime_error("SVD failed for a Berry overlap.");
    }
    minSingularValue = singularValues.min();
    checkOverlap(minSingularValue);
    return left * right.t();
}

bool nearCut(double phase) {
    return std::abs(phase) >= 0.9 * pi;
}

void averageDegenerateShares(arma::vec& shares, const arma::vec& energies, double tolerance) {
    arma::uword first = 0;
    while (first < energies.n_elem) {
        arma::uword end = first + 1;
        while (end < energies.n_elem && energies(end) - energies(end - 1) <= tolerance) {
            ++end;
        }
        if (end - first > 1) {
            auto block = shares.subvec(first, end - 1);
            const double mean = arma::mean(block);
            block.fill(mean);
        }
        first = end;
    }
}

} // namespace

arma::cx_double normalizedBerryLink(arma::cx_double overlap) {
    const auto magnitude = std::abs(overlap);
    checkOverlap(magnitude);
    return overlap / magnitude;
}

arma::cx_double determinantBerryLink(const arma::cx_mat& from, const arma::cx_mat& to) {
    double smallest;
    return normalizedBerryLink(arma::det(unitaryOverlap(from, to, smallest)));
}

TriangleWilsonPhases triangleWilsonPhases(
    const std::array<const arma::cx_mat*, 3>& frames,
    const std::array<const arma::vec*, 3>& energies,
    double degeneracyTolerance) {
    if (!std::isfinite(degeneracyTolerance) || degeneracyTolerance < 0) {
        throw std::invalid_argument("Degeneracy tolerance must be finite and nonnegative.");
    }
    if (frames[0] == nullptr || frames[0]->is_empty() || frames[0]->n_cols > frames[0]->n_rows) {
        throw std::invalid_argument("Expected nonempty orthonormal triangle frames.");
    }

    const arma::uword numBands = frames[0]->n_cols;
    for (std::size_t vertex = 0; vertex < frames.size(); ++vertex) {
        if (frames[vertex] == nullptr || frames[vertex]->n_cols != numBands ||
            frames[vertex]->n_rows != frames[0]->n_rows || !frames[vertex]->is_finite() ||
            energies[vertex] == nullptr || energies[vertex]->n_elem != numBands ||
            !energies[vertex]->is_finite() || !energies[vertex]->is_sorted()) {
            throw std::invalid_argument("Triangle frames must have the same dimensions and sorted finite energies.");
        }
    }

    double smallestOverlap = 1;
    if (numBands == 1) {
        arma::cx_double loop{1, 0};
        for (std::size_t vertex = 0; vertex < frames.size(); ++vertex) {
            const auto& next = *frames[(vertex + 1) % frames.size()];
            const arma::cx_double overlap = arma::cdot(frames[vertex]->col(0), next.col(0));
            smallestOverlap = std::min(smallestOverlap, std::abs(overlap));
            loop *= normalizedBerryLink(overlap);
        }
        const double phase = -std::arg(loop);
        return {phase, arma::vec{phase}, smallestOverlap, nearCut(phase)};
    }

    std::array<arma::cx_mat, 3> links;
    for (std::size_t vertex = 0; vertex < frames.size(); ++vertex) {
        const auto& next = *frames[(vertex + 1) % frames.size()];
        double smallest;
        links[vertex] = unitaryOverlap(*frames[vertex], next, smallest);
        smallestOverlap = std::min(smallestOverlap, smallest);
    }

    const arma::cx_mat loop = links[0] * links[1] * links[2];
    arma::cx_mat modes, triangular;
    // The loop is unitary, so Schur gives orthonormal Wilson eigenvectors.
    if (!arma::schur(modes, triangular, loop)) {
        throw std::runtime_error("Schur decomposition failed for the triangle Wilson loop.");
    }
    arma::vec eigenphases(numBands);
    bool nearBranchCut = false;
    for (arma::uword band = 0; band < numBands; ++band) {
        eigenphases(band) = -std::arg(triangular(band, band));
        nearBranchCut = nearBranchCut || nearCut(eigenphases(band));
    }
    const double phaseSum = arma::accu(eigenphases);
    const double total = std::remainder(phaseSum, 2 * pi);
    nearBranchCut = nearBranchCut || nearCut(total) || std::abs(phaseSum - total) > pi;

    arma::vec shares(numBands, arma::fill::zeros);
    for (std::size_t vertex = 0; vertex < frames.size(); ++vertex) {
        arma::vec localShares = arma::square(arma::abs(modes)) * eigenphases;
        averageDegenerateShares(localShares, *energies[vertex], degeneracyTolerance);
        shares += localShares;
        if (vertex + 1 < frames.size()) {
            // Moving the basepoint conjugates the loop by this link's adjoint.
            modes = links[vertex].t() * modes;
        }
    }
    shares /= static_cast<double>(frames.size());
    return {total, std::move(shares), smallestOverlap, nearBranchCut};
}

} // namespace quantumgeom
