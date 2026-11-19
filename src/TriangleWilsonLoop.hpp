/*
 * Project Name: QuantumGeom
 * Copyright (C) 2026 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef QUANTUMGEOM_TRIANGLE_WILSON_LOOP_HPP
#define QUANTUMGEOM_TRIANGLE_WILSON_LOOP_HPP

#include <array>
#include <armadillo>

namespace quantumgeom {

struct TriangleWilsonPhases {
    double total;
    arma::vec shares;
    double minSingularValue;
    bool nearBranchCut;
};

// Return the unit complex overlap for one band or the determinant of the
// unitary overlap between two orthonormal frames of the same band group.
arma::cx_double normalizedBerryLink(arma::cx_double overlap);
arma::cx_double determinantBerryLink(const arma::cx_mat& from, const arma::cx_mat& to);

// Each vertex provides an orthonormal frame for the same band group, with
// columns ordered by ascending eigenenergy. Phases are in radians and follow
// QuantumGeom's minus-argument convention for the given vertex order.
// Shares use the principal Wilson eigenphases, so their sum agrees with the
// determinant phase modulo 2*pi. An unresolved overlap raises an exception.
TriangleWilsonPhases triangleWilsonPhases(
    const std::array<const arma::cx_mat*, 3>& frames,
    const std::array<const arma::vec*, 3>& energies,
    double degeneracyTolerance);

} // namespace quantumgeom

#endif
