cmake_minimum_required(VERSION 3.16)

project(asteroids LANGUAGES CXX)

add_executable(asteroids
  src/main.cc
  src/asteroids.cc
)

target_compile_features(asteroids PRIVATE cxx_std_20)
set_target_properties(asteroids PROPERTIES
  CXX_EXTENSIONS OFF
)

target_include_directories(asteroids PRIVATE include/asteroids)

find_package(SDL2 QUIET)
if(EMSCRIPTEN)
  target_compile_definitions(asteroids PRIVATE AST_USE_SDL2=1)
  target_compile_options(asteroids PRIVATE "-sUSE_SDL=2")
  target_link_options(asteroids PRIVATE "-sUSE_SDL=2")
  target_link_options(asteroids PRIVATE "--shell-file" "${CMAKE_SOURCE_DIR}/shell.html")
  set_target_properties(asteroids PROPERTIES OUTPUT_NAME "index")
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
elseif(SDL2_FOUND)
  target_compile_definitions(asteroids PRIVATE AST_USE_SDL2=1)
  if(TARGET SDL2::SDL2)
    target_link_libraries(asteroids PRIVATE SDL2::SDL2)
  else()
    target_include_directories(asteroids PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(asteroids PRIVATE ${SDL2_LIBRARIES})
  endif()
endif()
