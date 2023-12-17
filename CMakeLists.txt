# The MIT License (MIT)
#
# Copyright (c)2016 Olivier Soares
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


cmake_minimum_required(VERSION 3.5)

project(jik)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("Making ${CMAKE_BUILD_TYPE}")

# Version
set(JIK_VERSION_MAJOR 0)
set(JIK_VERSION_MINOR 1)
set(JIK_VERSION ${JIK_VERSION_MAJOR}.${JIK_VERSION_MINOR})
add_definitions(-DJIK_VERSION=${JIK_VERSION})

# Default flags
set(CMAKE_CXX_FLAGS_SHARED "-Wall -Wpedantic -fPIC -DPIC -std=c++1y -fno-strict-aliasing -funroll-loops -I${CMAKE_CURRENT_SOURCE_DIR}")
if(WIN32)
  set(CMAKE_CXX_FLAGS_SHARED "${CMAKE_CXX_FLAGS_SHARED} -DWIN")
elseif(APPLE)
  set(CMAKE_CXX_FLAGS_SHARED "${CMAKE_CXX_FLAGS_SHARED} -DDARWIN")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS_SHARED "${CMAKE_CXX_FLAGS_SHARED} -DLINUX")
endif()

# Debug and release flags
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_SHARED} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_SHARED} -DNDEBUG -O3 -ffast-math -ftree-vectorize")

# Lib prefix and suffix
if(WIN32)
  set(LIB_PREFIX)
  set(LIB_SUFFIX     .lib)
  set(LIB_SUFFIX_DYN .dll)
elseif(APPLE)
  set(LIB_PREFIX     lib   )
  set(LIB_SUFFIX     .a    )
  set(LIB_SUFFIX_DYN .dylib)
else()
  set(LIB_PREFIX     lib)
  set(LIB_SUFFIX     .a )
  set(LIB_SUFFIX_DYN .so)
endif()

# Subdirectories
add_subdirectory(core)
add_subdirectory(recurrent)
add_subdirectory(sandbox)

# Add cpplint target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
