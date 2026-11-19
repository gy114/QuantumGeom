/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef MNTEPROJECT_WRITESURFACETOVTK_HPP
#define MNTEPROJECT_WRITESURFACETOVTK_HPP


#include "CGAL_dataStructureDefinitions.hpp"

//void write_to_vtp_SurfaceMesh(const std::string& filename, const Surface_mesh& mesh);
//void write_to_vtp_HandleSurfaceMesh(const std::string& filename, const HandleSurfaceMesh& mesh, Iso_cuboid cuboid, P3_Delaunay& T);
//void write_to_vtp_PeriodicSurfaceMesh(const std::string& filename, const PeriodicSurfaceMesh & mesh, const Iso_cuboid cuboid, const std::vector<double> InfoList);
//void write_to_vtu_P3(const std::string& filename, P3_Delaunay& T, const Iso_cuboid& cuboid, arma::uword iBand);
void write_to_vtu_P3(const std::string& filename, P3_Delaunay& T, const Iso_cuboid& cuboid, arma::uword iBand, bool toXYZ, arma::mat B);
void writeParallelepipedEdgesVtu(arma::vec3 b1, arma::vec3 b2, arma::vec3 b3, const std::string& edgesFilename);



#endif //MNTEPROJECT_WRITESURFACETOVTK_HPP
