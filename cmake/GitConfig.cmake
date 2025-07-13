# GitConfig.cmake - Handle Git configuration for FetchContent
# This module configures Git settings to avoid common issues with FetchContent

# Set Git configuration to handle ownership and checkout issues
find_package(Git REQUIRED)

# Configure Git to handle safe directories
execute_process(
    COMMAND ${GIT_EXECUTABLE} config --global --add safe.directory "*"
    RESULT_VARIABLE GIT_CONFIG_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

# Disable detached HEAD warnings
execute_process(
    COMMAND ${GIT_EXECUTABLE} config --global advice.detachedHead false
    RESULT_VARIABLE GIT_CONFIG_RESULT2
    OUTPUT_QUIET
    ERROR_QUIET
)

# Set longer timeout for Git operations
execute_process(
    COMMAND ${GIT_EXECUTABLE} config --global http.lowSpeedLimit 1000
    RESULT_VARIABLE GIT_CONFIG_RESULT3
    OUTPUT_QUIET
    ERROR_QUIET
)

execute_process(
    COMMAND ${GIT_EXECUTABLE} config --global http.lowSpeedTime 300
    RESULT_VARIABLE GIT_CONFIG_RESULT4
    OUTPUT_QUIET
    ERROR_QUIET
)

message(STATUS "Git configuration applied for FetchContent operations") 