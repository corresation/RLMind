cmake_minimum_required(VERSION 3.20)
project(RLMind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Release C runtime in every config, Python and the Rust lib both use it
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

find_package(Python 3.11 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
execute_process(
	COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
	OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(
	COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.base_prefix)"
	OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTHON_HOME)
file(TO_CMAKE_PATH "${PYTHON_HOME}" PYTHON_HOME)

# RocketSim v3 (Rust) plus the C functions in sim/rocketsim/bridge.rs, built by cargo
file(GLOB_RECURSE ROCKETSIM_SRC CONFIGURE_DEPENDS "RocketSim/src/*.rs")
set(BRIDGE_LIB "${CMAKE_BINARY_DIR}/cargo/release/rsbridge.lib")
add_custom_command(
	OUTPUT "${BRIDGE_LIB}"
	COMMAND cargo build --release --locked --manifest-path "${CMAKE_SOURCE_DIR}/sim/rocketsim/Cargo.toml" --target-dir "${CMAKE_BINARY_DIR}/cargo"
	DEPENDS ${ROCKETSIM_SRC} RocketSim/Cargo.toml sim/rocketsim/bridge.rs sim/rocketsim/Cargo.toml sim/rocketsim/Cargo.lock
	USES_TERMINAL)
add_custom_target(rsbridge DEPENDS "${BRIDGE_LIB}")

# The game side, as the python module sim/_sim.pyd (the last config built is the one used)
file(GLOB_RECURSE SIM_SRC CONFIGURE_DEPENDS "sim/*.cpp" "sim/*.h")
list(REMOVE_ITEM SIM_SRC "${CMAKE_SOURCE_DIR}/sim/RLMind.cpp")
nanobind_add_module(_sim NOMINSIZE ${SIM_SRC})
add_dependencies(_sim rsbridge)
target_link_libraries(_sim PRIVATE "${BRIDGE_LIB}" ntdll userenv ws2_32 dbghelp)
set_target_properties(_sim PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/sim")

# RLMind.exe, starts Python and runs main.py
add_executable(RLMind sim/RLMind.cpp)
target_link_libraries(RLMind PRIVATE Python::Python)
target_compile_definitions(RLMind PRIVATE
	"PYTHON_HOME=L\"${PYTHON_HOME}\""
	"PYTHON_EXE=L\"${Python_EXECUTABLE}\"")
add_dependencies(RLMind _sim)

# Copy the Python DLLs next to RLMind.exe so it starts without them being on PATH
file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/python*.dll")
add_custom_command(TARGET RLMind POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PYTHON_DLLS} "$<TARGET_FILE_DIR:RLMind>")
