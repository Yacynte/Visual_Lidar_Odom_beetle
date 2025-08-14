cmake_minimum_required(VERSION 3.10)
project(VisualOdometry)


include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(json)


# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Check if cross-compiling
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Cross-compiling for Raspberry Pi ARM64")
else()
    message(STATUS "Building for x86_64 PC")
endif()

# Ensure C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV 4 REQUIRED)

find_package(PCL REQUIRED)

find_package(Ceres REQUIRED)

if(NOT OpenCV_FOUND OR NOT PCL_FOUND)
    message(FATAL_ERROR "OpenCV or PCL not found. Please install OpenCV.")
endif()


# Build a shared library (No main function in shared lib)
add_library(VisualOdometry SHARED
    src/vo_process.cpp
    src/visualodom.cpp
    src/vo_ceres.cpp # Include the Ceres optimization implementation
    # visual odometry implementation
)

# Include directories
# include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} include /usr/local/include)
target_include_directories( VisualOdometry PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/include
)
# Specify the include directories for the library
# add_definitions(${PCL_DEFINITIONS})

# Link libraries
target_link_libraries(VisualOdometry
    pthread
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json # Link against the nlohmann_json library
    ${PCL_LIBRARIES} # Link against PCL libraries
    Ceres::ceres # Link against Ceres Solver
)

# Set output directory for the shared library (optional but recommended)
set_target_properties(VisualOdometry PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../../Assets/Plugins" # path to add library in Unity
    # ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../../Assets/Plugins" # path to add library in Unity
    # RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../../Assets/Plugins" # path to add library in Unity
    # Set the library name without the platform-specific prefix/suffix
    # For example, the library will be named 'VisualOdometry.so'
    PREFIX "" # Remove the default 'lib' prefix on Linux
    SUFFIX ".so"
)


# to build the executable for testing purposes
add_executable(VisualOdometry_exe
                src/main.cpp # Assuming you have a separate main.cpp for testing
                src/vo_process.cpp
                src/vo_ceres.cpp # Include the Ceres optimization implementation
                )

# Build a separate executable for TCP communication
add_executable(VisualOdometry_tcp
                src/visualodom_tcp.cpp # TCP communication implementation
                src/vo_ceres.cpp # Include the Ceres optimization implementation
                )

add_executable(icp_pcd
                src/pcd_pose.cpp # ICP PCD processing implementation
                )

# Add pcl executable for testing purposes
target_link_libraries(icp_pcd
                        ${PCL_LIBRARIES}
                        ${OpenCV_LIBS}
                        pthread
                        # VisualOdometry # Link against the shared library
                    )

target_link_libraries(VisualOdometry_exe
                        nlohmann_json::nlohmann_json
                        pthread
                        ${OpenCV_LIBS}
                        VisualOdometry # Link against the shared library
                        Ceres::ceres # Link against Ceres Solver
                        )

target_link_libraries(VisualOdometry_tcp
                        nlohmann_json::nlohmann_json
                        pthread
                        ${OpenCV_LIBS}
                        VisualOdometry # Link against the shared library
                        Ceres::ceres # Link against Ceres Solver
                        )

# Link it to your library/executables
# target_link_libraries(VisualOdometry PRIVATE nlohmann_json::nlohmann_json)
# target_link_libraries(VisualOdometry_exe PRIVATE nlohmann_json::nlohmann_json)
# target_link_libraries(VisualOdometry_tcp PRIVATE nlohmann_json::nlohmann_json)