# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

IF(NOT ${WITH_PSLIB_MPI})
  return()
ENDIF(NOT ${WITH_PSLIB_MPI})

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with PSLIB_MPI in Paddle yet."
        "Force WITH_PSLIB_MPI=OFF")
    SET(WITH_PSLIB_MPI OFF CACHE STRING "Disable PSLIB_MPI package in Windows and MacOS" FORCE)
    return()
ENDIF()

INCLUDE(ExternalProject)

SET(PSLIB_MPI_PROJECT       "extern_pslib_mpi")
IF((NOT DEFINED PSLIB_MPI_NAME) OR (NOT DEFINED PSLIB_MPI_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(PSLIB_MPI_VER "0.1.0" CACHE STRING "" FORCE)
  SET(PSLIB_MPI_NAME "pslib_mpi" CACHE STRING "" FORCE)
  SET(PSLIB_MPI_URL "http://10.199.242.50:8113/${PSLIB_MPI_NAME}.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "PSLIB_MPI_NAME: ${PSLIB_MPI_NAME}, PSLIB_MPI_URL: ${PSLIB_MPI_URL}")
SET(PSLIB_MPI_SOURCE_DIR    "${THIRD_PARTY_PATH}/pslib_mpi")
SET(PSLIB_MPI_DOWNLOAD_DIR  "${PSLIB_MPI_SOURCE_DIR}/src/${PSLIB_MPI_PROJECT}")
SET(PSLIB_MPI_DST_DIR       "pslib_mpi")
SET(PSLIB_MPI_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(PSLIB_MPI_INSTALL_DIR   ${PSLIB_MPI_INSTALL_ROOT}/${PSLIB_MPI_DST_DIR})
SET(PSLIB_MPI_ROOT          ${PSLIB_MPI_INSTALL_DIR})
SET(PSLIB_MPI_INC_DIR       ${PSLIB_MPI_ROOT}/include)
SET(PSLIB_MPI_LIB_DIR       ${PSLIB_MPI_ROOT}/lib)
SET(PSLIB_MPI_LIB           ${PSLIB_MPI_LIB_DIR}/libmpi.so)
SET(PSLIB_MPICXX_LIB           ${PSLIB_MPI_LIB_DIR}/libmpi_cxx.so)
SET(PSLIB_MPIOPENPAL_LIB           ${PSLIB_MPI_LIB_DIR}/libopen-pal.so)
SET(PSLIB_MPIOPENRTE_LIB           ${PSLIB_MPI_LIB_DIR}/libopen-rte.so)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${PSLIB_MPI_ROOT}/lib")

INCLUDE_DIRECTORIES(${PSLIB_MPI_INC_DIR})

FILE(WRITE ${PSLIB_MPI_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(PSLIB_MPI)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${PSLIB_MPI_NAME}/include ${PSLIB_MPI_NAME}/lib \n"
  "        DESTINATION ${PSLIB_MPI_DST_DIR})\n")

ExternalProject_Add(
    ${PSLIB_MPI_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${PSLIB_MPI_SOURCE_DIR}
    DOWNLOAD_DIR          ${PSLIB_MPI_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${PSLIB_MPI_URL} -c -q -O ${PSLIB_MPI_NAME}.tar.gz
                          && tar zxvf ${PSLIB_MPI_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${PSLIB_MPI_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${PSLIB_MPI_INSTALL_ROOT}
)

ADD_LIBRARY(pslib_mpi SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib_mpi PROPERTY IMPORTED_LOCATION ${PSLIB_MPI_LIB})
ADD_DEPENDENCIES(pslib_mpi ${PSLIB_MPI_PROJECT})
LIST(APPEND external_project_dependencies pslib_mpi)
ADD_LIBRARY(pslib_mpicxx SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib_mpicxx PROPERTY IMPORTED_LOCATION ${PSLIB_MPICXX_LIB})
ADD_DEPENDENCIES(pslib_mpicxx ${PSLIB_MPI_PROJECT})
LIST(APPEND external_project_dependencies pslib_mpicxx)
ADD_LIBRARY(pslib_mpiopenpal SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib_mpiopenpal PROPERTY IMPORTED_LOCATION ${PSLIB_MPIOPENPAL_LIB})
ADD_DEPENDENCIES(pslib_mpiopenpal ${PSLIB_MPI_PROJECT})
LIST(APPEND external_project_dependencies pslib_mpiopenpal)
ADD_LIBRARY(pslib_mpiopenrte SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib_mpiopenrte PROPERTY IMPORTED_LOCATION ${PSLIB_MPIOPENRTE_LIB})
ADD_DEPENDENCIES(pslib_mpiopenrte ${PSLIB_MPI_PROJECT})
LIST(APPEND external_project_dependencies pslib_mpiopenrte)
