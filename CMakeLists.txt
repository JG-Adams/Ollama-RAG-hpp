cmake_minimum_required(VERSION 3.16)
project(OLLAMA_RAGHPP VERSION 0.1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSION OFF)
set(CMAKE_CXX_STANDARD 11)

add_compile_options(-Wall -Wpedantic)

if (WIN32)
  if (MINGW)
    add_definitions(-D_WIN32_WINNT=0x0A00) # Ollama-hpp only work in window10+
  endif(MINGW)
    link_libraries(ws2_32)
endif (WIN32)

# output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(RAG example.cpp)
include_directories("include")
