cmake_minimum_required(VERSION 3.16)
project(OLLAMA_HPP VERSION 0.9.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSION OFF)
set(CMAKE_CXX_STANDARD 17)

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

add_executable(RAG main.cpp src/RAG.cpp)