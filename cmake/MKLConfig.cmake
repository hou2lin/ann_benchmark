# MKLConfig.cmake - Configuration file for Intel MKL on Ubuntu
# This file provides the MKL configuration that diskann is looking for

# Set MKL version
set(MKL_VERSION "2020.4.304")

# Set MKL include directories
set(MKL_INCLUDE_DIRS "/usr/include/mkl")

# Set MKL library directories
set(MKL_LIBRARY_DIRS "/usr/lib/x86_64-linux-gnu")

# Find MKL libraries
find_library(MKL_CORE_LIBRARY NAMES mkl_core PATHS ${MKL_LIBRARY_DIRS})
find_library(MKL_INTEL_LIBRARY NAMES mkl_intel_ilp64 PATHS ${MKL_LIBRARY_DIRS})
find_library(MKL_THREAD_LIBRARY NAMES mkl_intel_thread PATHS ${MKL_LIBRARY_DIRS})
find_library(MKL_DEF_LIBRARY NAMES mkl_def PATHS ${MKL_LIBRARY_DIRS})

# Set MKL libraries
if(MKL_CORE_LIBRARY AND MKL_INTEL_LIBRARY AND MKL_THREAD_LIBRARY)
    set(MKL_LIBRARIES ${MKL_INTEL_LIBRARY} ${MKL_THREAD_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_DEF_LIBRARY})
    set(MKL_FOUND TRUE)
    message(STATUS "MKL found: ${MKL_LIBRARIES}")
else()
    set(MKL_FOUND FALSE)
    message(STATUS "MKL libraries not found, will use system math libraries")
endif()

# Create MKL::MKL target for diskann
if(MKL_FOUND)
    # Create the MKL::MKL target that diskann expects
    add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}"
    )
    message(STATUS "MKL::MKL target created successfully")
else()
    message(STATUS "MKL not found, will use system math libraries")
endif() 