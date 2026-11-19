cmake_minimum_required(VERSION 3.16)
project(QuantumGeom VERSION 0.1.0)

configure_file(version.h.in version.h @ONLY)

set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CGAL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    IOXML
    FiltersCore
    FiltersGeometry
)

# Share the public numerical implementation between the binding and native tests.
add_library(quantumgeom_core STATIC
    src/Bulk.cpp
    src/Chain.cpp
    src/FermiSurface.cpp
    src/FermiSurfaceExtraction.cpp
    src/TriangleWilsonLoop.cpp
    src/WriteToVTK.cpp
)
set_target_properties(quantumgeom_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(quantumgeom_core PUBLIC cxx_std_17)
target_include_directories(quantumgeom_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${ARMADILLO_INCLUDE_DIRS}
)
target_link_libraries(quantumgeom_core PUBLIC
    ${ARMADILLO_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    CGAL::CGAL
    OpenMP::OpenMP_CXX
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::IOXML
    VTK::FiltersCore
    VTK::FiltersGeometry
)
if(APPLE)
    target_link_libraries(quantumgeom_core PUBLIC c++)
endif()

pybind11_add_module(quantumGeom MODULE src/pybind_main.cpp)
target_include_directories(quantumGeom PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(quantumGeom PRIVATE quantumgeom_core)

option(QUANTUMGEOM_BUILD_TESTS "Build native regression tests" OFF)
if(QUANTUMGEOM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install into the selected Python interpreter unless a packager overrides it.
if(NOT DEFINED PYTHON_SITE_PACKAGES)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE python_site_status
    )
    if(NOT python_site_status EQUAL 0 OR PYTHON_SITE_PACKAGES STREQUAL "")
        message(FATAL_ERROR "Could not determine Python site-packages. Set PYTHON_SITE_PACKAGES explicitly.")
    endif()
endif()

message(STATUS "Installing quantumGeom into: ${PYTHON_SITE_PACKAGES}")
install(TARGETS quantumGeom LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}")
