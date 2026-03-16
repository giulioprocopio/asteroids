cmake_minimum_required(VERSION 3.16)

project(asteroids LANGUAGES CXX)

add_executable(asteroids
  main.cc
  asteroids.cc
)

target_compile_features(asteroids PRIVATE cxx_std_20)
set_target_properties(asteroids PROPERTIES
  CXX_EXTENSIONS OFF
)

find_package(SDL2 QUIET)
if(SDL2_FOUND)
  target_compile_definitions(asteroids PRIVATE AST_USE_SDL2=1)
  if(TARGET SDL2::SDL2)
    target_link_libraries(asteroids PRIVATE SDL2::SDL2)
  else()
    target_include_directories(asteroids PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(asteroids PRIVATE ${SDL2_LIBRARIES})
  endif()
else()
  message(WARNING "SDL2 not found: building without animated window support")
endif()

if(MSVC)
  target_compile_options(asteroids PRIVATE /W4)
else()
  target_compile_options(asteroids PRIVATE -Wall -Wextra -Wpedantic)
endif()
