cmake_minimum_required(VERSION 3.20)

# set the project name
project(StamFluidSolver)

set(INCLUDES_DIR include)
set(INCLUDES glad glm-master)
set(LIBRARIES_DIR libs)
set(LIBRARIES glfw)
set(SOURCES_DIR src)

# Set binary directory at top level of build
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Include library headers
foreach(INC ${INCLUDES})
    include_directories("${INCLUDES_DIR}/${INC}/include")
endforeach(INC)

# Link libraries in lib
foreach(LIBRARY ${LIBRARIES})
    add_subdirectory("${LIBRARIES_DIR}/${LIBRARY}")
    include_directories("${LIBRARIES_DIR}/${LIBRARY}/include")
endforeach(LIBRARY)
#target_link_libraries(StamFluidSolver PUBLIC ${LIBRARIES})

# OpenGL
find_package(OpenGL REQUIRED)
#target_link_libraries(StamFluidSolver PUBLIC OpenGL::GL)

# Link threads library
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(StamFluidSolver PUBLIC Threads::Threads)

# Subdirectories
add_subdirectory(src)
