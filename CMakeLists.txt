cmake_minimum_required(VERSION 3.16)
project(cheapdan LANGUAGES C VERSION 0.1.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ---- Options ----------------------------------------------------------------
option(CHEAPDAN_BUILD_TESTS      "Build test suite"   ON)
option(CHEAPDAN_BUILD_BENCHMARKS "Build benchmarks"   ON)
option(CHEAPDAN_BUILD_EXAMPLES   "Build examples"     ON)

# ---- Compiler flags ---------------------------------------------------------
add_compile_options(
    $<$<C_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<C_COMPILER_ID:GNU,Clang>:-Wextra>
    $<$<C_COMPILER_ID:GNU,Clang>:-pedantic>
    $<$<AND:$<C_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<C_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-march=native>
)

# ---- FFTW3 ------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
endif()
if(NOT FFTW3_FOUND)
    find_package(FFTW3 QUIET)
endif()
if(NOT FFTW3_FOUND AND NOT FFTW3_LIBRARIES)
    message(FATAL_ERROR
        "FFTW3 not found. Install it with:\n"
        "  apt install libfftw3-dev    # Debian/Ubuntu\n"
        "  brew install fftw           # macOS\n"
        "Then re-run cmake.")
endif()

# ---- FetchContent: CHEAP ----------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    cheap
    GIT_REPOSITORY https://github.com/kevinfling/CHEAP
    GIT_TAG        master
)
set(CHEAP_BUILD_CPP  OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING    OFF CACHE BOOL "" FORCE)   # suppress CHEAP's own ctest targets
FetchContent_MakeAvailable(cheap)

# ---- INTERFACE library target -----------------------------------------------
add_library(cheapdan INTERFACE)
target_include_directories(cheapdan INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cheapdan INTERFACE cheap)

# ---- Subdirectories ---------------------------------------------------------
if(CHEAPDAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(CHEAPDAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(CHEAPDAN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---- Install ----------------------------------------------------------------
install(FILES
    include/cheapdan.h
    include/cheapdan_rts.h
    DESTINATION include
)
