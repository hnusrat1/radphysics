cmake_minimum_required(VERSION 3.16)
project(co60app CXX)

find_package(Geant4 11.4.3 EXACT REQUIRED)
include(${Geant4_USE_FILE})

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
add_executable(co60app co60app.cc ${sources})
target_include_directories(co60app PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(co60app PRIVATE cxx_std_17)
target_compile_options(co60app PRIVATE -Wall -Wextra -Wshadow)
target_link_libraries(co60app ${Geant4_LIBRARIES})
install(TARGETS co60app DESTINATION bin)
