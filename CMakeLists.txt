cmake_minimum_required(VERSION 3.10)
project(ELM_CPP VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to manually specify Eigen3 installation directory
set(EIGEN3_INSTALL_DIR "" CACHE PATH "Path to Eigen3 installation directory")

# Add Eigen3 search paths
if(EIGEN3_INSTALL_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${EIGEN3_INSTALL_DIR}")
endif()

# Add common Eigen3 installation paths (especially for Windows)
if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH 
        "C:/Program Files (x86)/Eigen3"
        "C:/Program Files/Eigen3"
        "C:/Eigen3"
        "$ENV{PROGRAMFILES}/Eigen3"
        "$ENV{PROGRAMFILES\(X86\)}/Eigen3"
    )
endif()

# Find Eigen3 library
find_package(Eigen3 3.3 NO_MODULE)

if(NOT Eigen3_FOUND)
    message(FATAL_ERROR 
        "Eigen3 not found. Please install Eigen3 or specify its location using:\n"
        "  cmake -DEIGEN3_INSTALL_DIR=\"C:/Program Files (x86)/Eigen3\" ..\n"
        "or set CMAKE_PREFIX_PATH to the Eigen3 installation directory.\n"
        "Download Eigen3 from: https://eigen.tuxfamily.org/")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add executable for training
add_executable(elm_train 
    train.cpp
)

# Add executable for prediction
add_executable(elm_predict 
    predict.cpp
)

# Link Eigen3 to both executables
target_link_libraries(elm_train Eigen3::Eigen)
target_link_libraries(elm_predict Eigen3::Eigen)

# Set output directories
set_target_properties(elm_train elm_predict PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation rules
install(TARGETS elm_train elm_predict
    RUNTIME DESTINATION bin
)

install(FILES elml.hpp
    DESTINATION include
)

# Enable warnings
if(MSVC)
    target_compile_options(elm_train PRIVATE /W4)
    target_compile_options(elm_predict PRIVATE /W4)
else()
    target_compile_options(elm_train PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(elm_predict PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optional: Build MEX files if MATLAB is found
option(BUILD_MEX "Build MATLAB MEX files" OFF)

if(BUILD_MEX)
    find_package(Matlab COMPONENTS MX_LIBRARY)
    
    if(Matlab_FOUND)
        matlab_add_mex(NAME mexElmTrain SRC mexElmTrain.cpp)
        matlab_add_mex(NAME mexElmPredict SRC mexElmPredict.cpp)
        
        target_link_libraries(mexElmTrain Eigen3::Eigen)
        target_link_libraries(mexElmPredict Eigen3::Eigen)
    else()
        message(WARNING "MATLAB not found. MEX files will not be built.")
    endif()
endif()

