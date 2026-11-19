#include "FermiSurfaceExtraction.hpp"

#include <cstddef>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Point = K_exact::Point_3;
using Exact = K_exact::FT;
using Polygons = std::vector<std::vector<std::size_t>>;

Polyhedron make_surface(const std::vector<Point>& points, const Polygons& faces) {
    Polyhedron surface;
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, faces, surface);
    surface.normalize_border();
    std::size_t id = 0;
    for (auto vertex = surface.vertices_begin(); vertex != surface.vertices_end(); ++vertex) {
        vertex->id() = id++;
    }
    return surface;
}

Polyhedron make_periodic_plane(bool leave_hole) {
    constexpr std::size_t steps = 4;
    constexpr std::size_t stride = steps + 1;
    const Exact half = Exact(1) / 2;
    const Exact y = Exact(3) / 4;
    std::vector<Point> points;
    for (std::size_t x = 0; x <= steps; ++x) {
        for (std::size_t z = 0; z <= steps; ++z) {
            points.emplace_back(Exact(x) / Exact(steps) - half, y, Exact(z) / Exact(steps) - half);
        }
    }

    // A flat tile that closes periodically in x and z. Its y coordinate puts
    // all vertices, including every seam copy, outside the central cuboid.
    Polygons faces;
    for (std::size_t x = 0; x < steps; ++x) {
        for (std::size_t z = 0; z < steps; ++z) {
            const std::size_t i = x * stride + z;
            if (!leave_hole || x != 1 || z != 1) {
                faces.push_back({i, i + stride, i + 1});
            }
            faces.push_back({i + stride, i + stride + 1, i + 1});
        }
    }
    return make_surface(points, faces);
}

void check_seam_pairs(const Polyhedron& surface, const PeriodicSurfaceConnectivity& connectivity,
                      std::size_t expected) {
    if (connectivity.seamPairs.size() != expected) {
        throw std::runtime_error("Unexpected number of retained seam pairs");
    }

    std::vector<Point> points(surface.size_of_vertices());
    for (auto vertex = surface.vertices_begin(); vertex != surface.vertices_end(); ++vertex) {
        points.at(vertex->id()) = vertex->point();
    }

    std::set<std::pair<Point, Point>> faceOrientedBorderEdges;
    for (auto face = surface.facets_begin(); face != surface.facets_end(); ++face) {
        const auto first = face->facet_begin();
        auto edge = first;
        do {
            if (edge->opposite()->is_border()) {
                faceOrientedBorderEdges.emplace(edge->opposite()->vertex()->point(), edge->vertex()->point());
            }
            ++edge;
        } while (edge != first);
    }

    for (const auto& seam : connectivity.seamPairs) {
        const auto& source = points.at(seam.source);
        const auto& target = points.at(seam.target);
        const K_exact::Vector_3 shift(static_cast<double>(seam.latticeShift[0]),
                                      static_cast<double>(seam.latticeShift[1]),
                                      static_cast<double>(seam.latticeShift[2]));
        if (shift == CGAL::NULL_VECTOR || faceOrientedBorderEdges.erase({source, target}) != 1 ||
            faceOrientedBorderEdges.erase({target - shift, source - shift}) != 1) {
            throw std::runtime_error("A seam must record its face direction and signed shift to the reverse copy");
        }
    }
    if (faceOrientedBorderEdges.size() != connectivity.unmatchedBorderHalfedges) {
        throw std::runtime_error("Each matched halfedge must occur in exactly one seam pair");
    }
}

void test_periodic_plane() {
    const auto surface = make_periodic_plane(false);
    const auto connectivity = buildPeriodicSurfaceConnectivity(surface);
    if (connectivity.unmatchedBorderHalfedges != 0) {
        throw std::runtime_error("The periodic plane seams should all pair");
    }
    check_seam_pairs(surface, connectivity, 8);

    const auto& representatives = connectivity.vertexRepresentative;
    if (representatives.size() != surface.size_of_vertices()) {
        throw std::runtime_error("Every planar vertex must have a representative");
    }

    std::vector<Point> points(representatives.size());
    for (auto vertex = surface.vertices_begin(); vertex != surface.vertices_end(); ++vertex) {
        points.at(vertex->id()) = vertex->point();
    }

    const Exact half = Exact(1) / 2;
    std::set<std::size_t> distinct_representatives;
    for (std::size_t id = 0; id < representatives.size(); ++id) {
        const auto& p = points.at(id);
        const auto representative = representatives.at(id);
        const auto& q = points.at(representative);
        const Exact dx = CGAL::abs(p.x() - q.x());
        const Exact dz = CGAL::abs(p.z() - q.z());
        if ((dx != 0 && dx != 1) || (dz != 0 && dz != 1) ||
            p.y() != q.y() || p.y() <= half || representatives.at(representative) != representative) {
            throw std::runtime_error("The representative must be an existing periodic image");
        }
        distinct_representatives.insert(representative);
    }
    if (distinct_representatives.size() != 16) {
        throw std::runtime_error("Expected 16 vertices after joining both periodic seams");
    }
}

void test_open_borders() {
    const auto with_hole = make_periodic_plane(true);
    const auto with_hole_connectivity = buildPeriodicSurfaceConnectivity(with_hole);
    if (with_hole_connectivity.unmatchedBorderHalfedges != 3) {
        throw std::runtime_error("A removed triangle must leave three unmatched halfedges");
    }
    check_seam_pairs(with_hole, with_hole_connectivity, 8);

    const Exact quarter = Exact(1) / 4;
    const auto same_winding = make_surface(
        {{0, 0, 0}, {quarter, 0, 0}, {0, quarter, 0},
         {0, 0, 1}, {quarter, 0, 1}, {0, quarter, 1}},
        {{0, 1, 2}, {3, 4, 5}});
    if (buildPeriodicSurfaceConnectivity(same_winding).unmatchedBorderHalfedges != 6) {
        throw std::runtime_error("Periodic border halfedges with the same direction must not pair");
    }

    const auto different_lifts = make_surface(
        {{0, 0, 0}, {quarter, 0, 0}, {0, quarter, 0},
         {0, 0, 1}, {1 + quarter, 0, 1}, {0, quarter, 1}},
        {{0, 1, 2}, {3, 5, 4}});
    if (buildPeriodicSurfaceConnectivity(different_lifts).unmatchedBorderHalfedges != 4) {
        throw std::runtime_error("Matching endpoints must also have the same relative lattice shift");
    }
}

void test_isolated_translated_triangle() {
    const Exact quarter = Exact(1) / 4;
    const auto surface = make_surface(
        {{0, 0, 0}, {quarter, 0, 0}, {0, quarter, 0}, {0, 0, quarter},
         {2, -1, 1}, {2 + quarter, -1, 1}, {2, -1 + quarter, 1}},
        {{0, 1, 3}, {0, 3, 2}, {1, 2, 3}, {4, 6, 5}});
    const auto connectivity = buildPeriodicSurfaceConnectivity(surface);
    if (connectivity.unmatchedBorderHalfedges != 0) {
        throw std::runtime_error("All edges of the isolated periodic face must pair");
    }
    check_seam_pairs(surface, connectivity, 3);
}

void test_canonical_coordinates() {
    const Exact half = Exact(1) / 2;
    const Exact quarter = Exact(1) / 4;
    const Point lifted(half, 3 + quarter, -2 - half);
    if (map_to_exact_cuboid(lifted) != Point(-half, quarter, -half)) {
        throw std::runtime_error("Canonicalization must use the half-open cuboid for any lattice offset");
    }
}

}  // namespace

int main() {
    try {
        test_periodic_plane();
        test_open_borders();
        test_isolated_translated_triangle();
        test_canonical_coordinates();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
