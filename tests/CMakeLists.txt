add_executable(test_berry_triangle_area test_berry_triangle_area.cpp)
target_link_libraries(test_berry_triangle_area PRIVATE quantumgeom_core)

foreach(basis identity scaled sheared)
    add_test(NAME berry_triangle_area_${basis} COMMAND test_berry_triangle_area ${basis})
endforeach()

add_executable(test_berry_refinement_selection test_berry_refinement_selection.cpp)
target_link_libraries(test_berry_refinement_selection PRIVATE quantumgeom_core)

function(add_refinement_count_test name candidates depth expected)
    add_test(NAME berry_refinement_${name}
        COMMAND test_berry_refinement_selection count ${candidates} ${depth} ${expected})
endfunction()

add_refinement_count_test(empty 0 100 0)
add_refinement_count_test(zero_percent 50 0 0)
add_refinement_count_test(negative_percent 50 -10 0)
add_refinement_count_test(above_100_percent 50 120 50)
add_refinement_count_test(small_all 50 100 50)
add_refinement_count_test(one_percent 100 1 1)
add_refinement_count_test(hundred_all 100 100 100)
add_refinement_count_test(half_150 150 50 75)
add_refinement_count_test(all_150 150 100 150)
add_refinement_count_test(fractional_count 51 50 25)
add_refinement_count_test(less_than_one_cell 50 1 0)
add_refinement_count_test(single_cell 1 100 1)
add_test(NAME berry_refinement_ranking COMMAND test_berry_refinement_selection ranking)

add_executable(test_chain_missing_counterpart test_chain_missing_counterpart.cpp)
target_link_libraries(test_chain_missing_counterpart PRIVATE quantumgeom_core)
add_test(NAME chain_missing_counterpart COMMAND test_chain_missing_counterpart)

add_executable(test_fermi_periodic_connectivity test_fermi_periodic_connectivity.cpp)
target_link_libraries(test_fermi_periodic_connectivity PRIVATE quantumgeom_core)
add_test(NAME fermi_periodic_connectivity COMMAND test_fermi_periodic_connectivity)

add_executable(test_triangle_wilson_loop test_triangle_wilson_loop.cpp)
target_link_libraries(test_triangle_wilson_loop PRIVATE quantumgeom_core)
add_test(NAME triangle_wilson_loop COMMAND test_triangle_wilson_loop)

add_executable(test_bulk_eigenvalue_mesh test_bulk_eigenvalue_mesh.cpp)
target_link_libraries(test_bulk_eigenvalue_mesh PRIVATE quantumgeom_core)
add_test(NAME bulk_eigenvalue_mesh COMMAND test_bulk_eigenvalue_mesh
    "${PROJECT_SOURCE_DIR}/examples/weyl_semi_tb.dat")
