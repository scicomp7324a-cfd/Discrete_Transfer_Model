cmake_minimum_required(VERSION 3.20)

project(DTM_V1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80-real)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    $ENV{HOME}/optix/optix-dev/include
)

set(OPTIX_SOURCES
    src/optix/GasBuilder.cu
    src/optix/IasBuilder.cu
    src/optix/ModuleBuilder.cu
    src/optix/PipelineBuilder.cu
    src/optix/ProgramGroupBuilder.cu
    src/optix/SbtBuilder.cu
    src/optix/LaunchManager.cu
)

set(RADIATION_SOURCES
    src/radiation/DataBuilder.cu
    src/radiation/ViewFactor.cu
    src/radiation/RadiativeFlux.cu
    src/radiation/Launcher.cu
)

add_executable(dtm_app
    src/main.cu
    ${OPTIX_SOURCES}
    ${RADIATION_SOURCES}
)

target_include_directories(dtm_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{HOME}/optix/optix-dev/include
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/devicePrograms.ptx
    COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx ${CMAKE_SOURCE_DIR}/kernels/devicePrograms.cu
        -I${CMAKE_SOURCE_DIR}/include
        -I$ENV{HOME}/optix/optix-dev/include
        -o ${CMAKE_BINARY_DIR}/devicePrograms.ptx
    DEPENDS
        ${CMAKE_SOURCE_DIR}/kernels/devicePrograms.cu
        ${CMAKE_SOURCE_DIR}/include/core/RayParams.hpp
        ${CMAKE_SOURCE_DIR}/include/core/RayGeometry.hpp
        ${CMAKE_SOURCE_DIR}/include/core/TriangleTraceHistory.hpp
    COMMENT "Compiling OptiX devicePrograms.cu to PTX"
)

add_custom_target(optix_ptx ALL
    DEPENDS ${CMAKE_BINARY_DIR}/devicePrograms.ptx
)

add_dependencies(dtm_app optix_ptx)

add_custom_command(
    TARGET dtm_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Output
)