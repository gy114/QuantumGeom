/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef MNTEPROJECT_CGAL_DATASTRUCTUREDEFINITIONS_HPP
#define MNTEPROJECT_CGAL_DATASTRUCTUREDEFINITIONS_HPP


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Periodic_3_triangulation_ds_vertex_base_3.h>
#include <CGAL/Periodic_3_triangulation_ds_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/make_periodic_3_mesh_3.h>
#include <CGAL/Periodic_3_mesh_triangulation_3.h>
#include <CGAL/Triangulation_utils_3.h>
//#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/facets_in_complex_3_to_triangle_mesh.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Surface_mesh.h>
//#include <CGAL/tetrahedron_soup_to_triangulation_3.h>
#include <CGAL/Tetrahedral_remeshing/Remeshing_triangulation_3.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/merge_border_vertices.h>
//#include <CGAL/copy_face_graph.h>
#include <armadillo>


#include <CGAL/number_type_config.h> // CGAL_PI




typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_exact_constructions_kernel K_exact;
typedef CGAL::Periodic_3_Delaunay_triangulation_traits_3<K> P3_Traits;
typedef CGAL::Periodic_3_triangulation_ds_vertex_base_3<> P3_VbDS;
typedef CGAL::Triangulation_vertex_base_3<P3_Traits, P3_VbDS >       P3_Vb;
typedef CGAL::Periodic_3_triangulation_ds_cell_base_3<> P3_CbDS;
typedef CGAL::Triangulation_cell_base_3<P3_Traits, P3_CbDS >       P3_Cb;

struct VertexData {
    arma::vec eigVal;
    K_exact::Point_3 exactPoint;
    std::vector<K_exact::FT> exactEnergy;
    bool isExact = false;
    arma::vec berryFlux;
};
struct CellData {
    std::vector<arma::vec> facetBerryFlux;
    // this contains the single-band berry flux for the facets opposit to vertices 0 -> 3.
};
typedef CGAL::Triangulation_vertex_base_with_info_3<VertexData, P3_Traits, P3_Vb > VbInfo; // Test to attach data to vertices.
typedef CGAL::Triangulation_cell_base_with_info_3<CellData, P3_Traits, P3_Cb > CbInfo; // Test to attach data to cells.
typedef CGAL::Triangulation_data_structure_3<VbInfo, CbInfo >                 TDS;
typedef CGAL::Periodic_3_Delaunay_triangulation_3<P3_Traits, TDS >               P3_Delaunay;
typedef P3_Delaunay::Point P3_Point;
typedef P3_Delaunay::Periodic_point PeriodicPoint;
typedef P3_Delaunay::Iso_cuboid Iso_cuboid;
typedef CGAL::Polyhedron_3<K_exact, CGAL::Polyhedron_items_with_id_3> Polyhedron;

//typedef CGAL::Surface_mesh<P3_Point> Surface_mesh;
//typedef CGAL::Surface_mesh<P3_Delaunay::Vertex_handle> HandleSurfaceMesh;
//typedef CGAL::Surface_mesh<PeriodicPoint> PeriodicSurfaceMesh;


//typedef CGAL::Labeled_mesh_domain_3<K>                      Periodic_mesh_domain;
// Triangulation
//typedef CGAL::Periodic_3_mesh_triangulation_3<Periodic_mesh_domain>::type Tr;
//typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr>                       C3t3;
// Criteria
//typedef CGAL::Mesh_criteria_3<Tr>                           Periodic_mesh_criteria;
// To avoid verbose function and named parameters call
namespace params = CGAL::parameters;

typedef CGAL::Tetrahedral_remeshing::Remeshing_triangulation_3<P3_Traits, TDS> RemeshingTriangulation;


#endif //MNTEPROJECT_CGAL_DATASTRUCTUREDEFINITIONS_HPP
