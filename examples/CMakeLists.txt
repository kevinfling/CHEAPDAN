foreach(ex example_1d example_2d)
    add_executable(${ex} ${ex}.c)
    target_link_libraries(${ex} PRIVATE cheapdan m)
endforeach()
