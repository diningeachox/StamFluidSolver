cmake_minimum_required(VERSION 3.20)

# set the project name
project(StamFluidSolver)

# add the executable
add_executable(StamFluidSolver main.cpp density.cpp density.h shader.cpp shader.h kernels.cu kernels.cuh)
# Add shader files
target_sources(StamFluidSolver PRIVATE textureshader.vs textureshader.fs)
# Add glad source file
target_sources(StamFluidSolver PRIVATE glad.c)

# Set built properties with CUDA
set_property(TARGET StamFluidSolver
                       PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set working directory for MSVC
#if (MSVC)
    #set_property(TARGET StamFluidSolver PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})

# Copy shader files to build directory
add_custom_command(
        TARGET StamFluidSolver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/textureshader.vs
                ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/textureshader.vs)

add_custom_command(
        TARGET StamFluidSolver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/textureshader.fs
                ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/textureshader.fs)

# Link libraries
target_link_libraries(StamFluidSolver PUBLIC ${LIBRARIES})

# OpenGL
target_link_libraries(StamFluidSolver PUBLIC OpenGL::GL)

# Threads
target_link_libraries(StamFluidSolver PUBLIC Threads::Threads)