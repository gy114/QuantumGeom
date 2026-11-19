/*
 * Project Name: QuantumGeom
 * Copyright (C) 2025 Gen Yin, Yin's Group, Georgetown University
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
//#include <vtkUnstructuredGridWriter.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkTetra.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkXMLUnstructuredGridWriter.h>
//#include <vtkPolyData.h>
#include <vtkPolygon.h>
#include <vtkLine.h>
//#include <vtkXMLPolyDataWriter.h>

#include "CGAL_dataStructureDefinitions.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*void write_to_vtp_SurfaceMesh(const std::string& filename, const Surface_mesh& mesh) {
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();

    // Transfer vertices to VTK points
    for (auto v : mesh.vertices()) {
        const P3_Point& p = mesh.point(v);
        points->InsertNextPoint(p.x(), p.y(), p.z());
    }

    // Transfer faces to VTK polygons
    for (auto f : mesh.faces()) {
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
        polygon->GetPointIds()->SetNumberOfIds(mesh.degree(f));
        // Add vertex indices to the polygon
        size_t i = 0;
        for (auto v : vertices_around_face(mesh.halfedge(f), mesh)) {
            polygon->GetPointIds()->SetId(i++, v.idx());
        }
        polygons->InsertNextCell(polygon);
    }

    polyData->SetPoints(points);
    polyData->SetPolys(polygons);

    // Write to VTK file
    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polyData);
    //writer->SetDataModeToBinary();
    writer->SetDataModeToAscii();
    writer->Write();
}*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*void write_to_vtp_HandleSurfaceMesh(const std::string& filename, const HandleSurfaceMesh& mesh, Iso_cuboid cuboid, P3_Delaunay& T) {
    vtkSmartPointer<vtkDoubleArray> pointDataArray = vtkSmartPointer<vtkDoubleArray>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();

    double dx = cuboid.xmax() - cuboid.xmin();
    double dy = cuboid.ymax() - cuboid.ymin();
    double dz = cuboid.zmax() - cuboid.zmin();

    pointDataArray->SetNumberOfComponents(1); // Only one component for double info
    pointDataArray->SetName("PointData");

    // Transfer vertices and their info to VTK points and pointDataArray
    for (auto v_idx : mesh.vertices()) {
        P3_Delaunay::Vertex_handle v = mesh.point(v_idx);
        PeriodicPoint pp = T.periodic_point(v);
        P3_Point p = pp.first;
        auto translation = pp.second;
        points->InsertNextPoint(p.x()+translation[0]*dx, p.y()+translation[1]*dy, p.z()+translation[2]*dz);
        double infoValue = v->info(); // Assuming info() returns a double
        pointDataArray->InsertNextValue(infoValue);
    }

    // Transfer faces to VTK polygons
    for (auto f : mesh.faces()) {
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
        polygon->GetPointIds()->SetNumberOfIds(mesh.degree(f));
        size_t i = 0;
        for (auto v : vertices_around_face(mesh.halfedge(f), mesh)) {
            polygon->GetPointIds()->SetId(i++, v.idx());
        }
        polygons->InsertNextCell(polygon);
    }

    polyData->SetPoints(points);
    polyData->GetPointData()->SetScalars(pointDataArray); // Add point data array to polyData
    polyData->SetPolys(polygons);

    // Write to VTK file
    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polyData);
    writer->SetDataModeToBinary();
    //writer->SetDataModeToAscii();
    writer->Write();
}*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*void write_to_vtp_PeriodicSurfaceMesh(const std::string& filename, const PeriodicSurfaceMesh & mesh, const Iso_cuboid cuboid, const std::vector<double> InfoList) {
    vtkSmartPointer<vtkDoubleArray> pointDataArray = vtkSmartPointer<vtkDoubleArray>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();


    pointDataArray->SetNumberOfComponents(1); // Only one component for double info
    pointDataArray->SetName("PointData");

    // Transfer vertices and their info to VTK points and pointDataArray
    for (auto v_idx : mesh.vertices()) {
        PeriodicPoint v = mesh.point(v_idx);
        P3_Point p = v.first;
        P3_Delaunay::Offset offset = v.second;
        points->InsertNextPoint(p.x(), p.y(), p.z());

        double infoValue = InfoList[v_idx]; // Assuming info() returns a double
        pointDataArray->InsertNextValue(infoValue);
    }

    // Transfer faces to VTK polygons
    for (auto f : mesh.faces()) {
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
        polygon->GetPointIds()->SetNumberOfIds(mesh.degree(f));
        size_t i = 0;
        for (auto v : vertices_around_face(mesh.halfedge(f), mesh)) {
            polygon->GetPointIds()->SetId(i++, v.idx());
        }
        polygons->InsertNextCell(polygon);
    }

    polyData->SetPoints(points);
    polyData->GetPointData()->SetScalars(pointDataArray); // Add point data array to polyData
    polyData->SetPolys(polygons);

    // Write to VTK file
    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polyData);
    writer->SetDataModeToBinary();
    //writer->SetDataModeToAscii();
    writer->Write();
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void write_to_vtu_P3(const std::string& filename, P3_Delaunay& T, const Iso_cuboid& cuboid, arma::uword iBand, bool toXYZ, arma::mat B) {

    if (!T.is_1_cover())
        T.convert_to_1_sheeted_covering();

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

    // PointData now has 1 component for the double info
    vtkSmartPointer<vtkDoubleArray> pointDataArray = vtkSmartPointer<vtkDoubleArray>::New();
    pointDataArray->SetNumberOfComponents(1); // Only one component for double info
    vtkSmartPointer<vtkDoubleArray> cellDataArray = vtkSmartPointer<vtkDoubleArray>::New();

    std::map<P3_Point, vtkIdType, K::Less_xyz_3> pointToIDs;

    double dx = cuboid.xmax() - cuboid.xmin();
    double dy = cuboid.ymax() - cuboid.ymin();
    double dz = cuboid.zmax() - cuboid.zmin();

    pointDataArray->SetName("PointData");
    cellDataArray->SetName("CellData");

    for (auto t = T.unique_cells_begin(); t != T.unique_cells_end(); ++t) {

        vtkSmartPointer<vtkTetra> vtk_tetra = vtkSmartPointer<vtkTetra>::New();

        for (int i = 0; i < 4; ++i) {
            auto pp = T.periodic_point(t, i);
            P3_Point canonical_point = pp.first;
            CGAL::Periodic_3_offset_3 offset = pp.second;

            double x = CGAL::to_double(canonical_point.x()) + offset[0] * dx;
            double y = CGAL::to_double(canonical_point.y()) + offset[1] * dy;
            double z = CGAL::to_double(canonical_point.z()) + offset[2] * dz;
            P3_Point actual_point(x, y, z);

            if (pointToIDs.find(actual_point) == pointToIDs.end()) {
                if (!toXYZ)
                    pointToIDs[actual_point] = points->InsertNextPoint(x, y, z);
                else   {
                    arma::vec p_123 = arma::vec{x, y, z};
                    arma::vec p_xyz = B*p_123;
                    pointToIDs[actual_point] = points->InsertNextPoint(p_xyz(0), p_xyz(1), p_xyz(2));
                }
                // Append point data
                VertexData data = t->vertex(i)->info(); // Retrieve the double info directly
                double vertex_data = data.eigVal(iBand);
                pointDataArray->InsertNextValue(vertex_data);
            }
            vtk_tetra->GetPointIds()->SetId(i, pointToIDs[actual_point]);
        }

        unstructuredGrid->InsertNextCell(vtk_tetra->GetCellType(), vtk_tetra->GetPointIds());

        // Append cell data
        double cell_info = 0;
        if (t->info().facetBerryFlux.size() == 4 && iBand < t->info().facetBerryFlux[0].n_elem &&
                iBand < t->info().facetBerryFlux[1].n_elem && iBand < t->info().facetBerryFlux[2].n_elem
                && iBand < t->info().facetBerryFlux[3].n_elem)  {
            cell_info = t->info().facetBerryFlux[0](iBand) + t->info().facetBerryFlux[1](iBand) + t->info().facetBerryFlux[2](iBand) + t->info().facetBerryFlux[3](iBand); // Retrieve the double info directly
        }
        cellDataArray->InsertNextValue(cell_info);
    }

    unstructuredGrid->SetPoints(points);
    unstructuredGrid->GetPointData()->SetScalars(pointDataArray);
    unstructuredGrid->GetCellData()->SetScalars(cellDataArray);

    // Write to *.vtu file
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(unstructuredGrid);
    writer->SetDataModeToBinary();
    writer->Write();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void writeParallelepipedEdgesVtu(arma::vec3 b1, arma::vec3 b2, arma::vec3 b3, const std::string& edgesFilename) {
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();

    // Centering the parallelepiped at the origin
    arma::vec3 half_b1 = b1 * 0.5, half_b2 = b2 * 0.5, half_b3 = b3 * 0.5;
    arma::vec3 center = half_b1 + half_b2 + half_b3; // This is the new center point.

// Now subtract the center from each corner to center the parallelepiped at the origin
    std::vector<arma::vec3> corners = {
            -center, // This will be (-0.5, -0.5, -0.5)
            half_b1 - half_b2 - half_b3,
            -half_b1 + half_b2 - half_b3,
            -half_b1 - half_b2 + half_b3,
            half_b1 + half_b2 - half_b3,
            half_b1 - half_b2 + half_b3,
            -half_b1 + half_b2 + half_b3,
            center // This will be (0.5, 0.5, 0.5)
    };

    // Add corner points to vtkPoints
    for (const auto& corner : corners) {
        points->InsertNextPoint(corner(0), corner(1), corner(2));
    }
    grid->SetPoints(points);

    // Define edges
    std::vector<std::pair<int, int>> edges = {
            {0, 1}, {1, 4}, {4, 2}, {2, 0},
            {0, 3}, {1, 5}, {4, 7}, {2, 6},
            {3, 5}, {5, 7}, {7, 6}, {3, 6}
    };

    // Add lines to the grid
    for (const auto& edge : edges) {
        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, edge.first);
        line->GetPointIds()->SetId(1, edge.second);
        grid->InsertNextCell(line->GetCellType(), line->GetPointIds());
    }

    // Write to a VTU file
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName(edgesFilename.c_str());
    writer->SetInputData(grid);
    // For binary output, uncomment the following line
    // writer->SetDataModeToBinary();
    writer->Write();
}
