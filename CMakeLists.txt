# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMSISCORE ${CMAKE_CURRENT_SOURCE_DIR}/extern/CMSIS_5/CMSIS/Core)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)
set(PICO_DEOPTIMIZED_DEBUG ON)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.0)
set(toolchainVersion 13_3_Rel1)
set(picotoolVersion 2.1.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(hello_dsp_q16 C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

add_executable(
    hello_dsp_q16
    hello_dsp_q16.c
)

# pull in common dependencies
target_link_libraries(hello_dsp_q16 pico_stdlib CMSISDSP)

target_include_directories(
    hello_dsp_q16
    PRIVATE
    ${CMSISCORE}/Include
)

# create map/bin/hex file etc.
pico_add_extra_outputs(hello_dsp_q16)

# add url via pico_set_program_url

add_subdirectory(extern/CMSIS_DSP)
