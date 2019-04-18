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

IF(NOT ${WITH_PSLIB_MLPARALLEL})
  return()
ENDIF(NOT ${WITH_PSLIB_MLPARALLEL})

IF(WIN32 OR APPLE)
    MESSAGE(WARNING
        "Windows or Mac is not supported with PSLIB_MLPARALLEL in Paddle yet."
        "Force WITH_PSLIB_MLPARALLEL=OFF")
    SET(WITH_PSLIB_MLPARALLEL OFF CACHE STRING "Disable PSLIB_MLPARALLEL package in Windows and MacOS" FORCE)
    return()
ENDIF()

INCLUDE(ExternalProject)

SET(PSLIB_MLPARALLEL_PROJECT       "extern_pslib_mlparallel")
IF((NOT DEFINED PSLIB_MLPARALLEL_NAME) OR (NOT DEFINED PSLIB_MLPARALLEL_URL))
  MESSAGE(STATUS "use pre defined download url")
  SET(PSLIB_MLPARALLEL_VER "0.1.0" CACHE STRING "" FORCE)
  SET(PSLIB_MLPARALLEL_NAME "pslib_mlparallel" CACHE STRING "" FORCE)
  SET(PSLIB_MLPARALLEL_URL "http://10.199.242.50:8113/${PSLIB_MLPARALLEL_NAME}.tar.gz" CACHE STRING "" FORCE)
ENDIF()
MESSAGE(STATUS "PSLIB_MLPARALLEL_NAME: ${PSLIB_MLPARALLEL_NAME}, PSLIB_MLPARALLEL_URL: ${PSLIB_MLPARALLEL_URL}")
SET(PSLIB_MLPARALLEL_SOURCE_DIR    "${THIRD_PARTY_PATH}/pslib_mlparallel")
SET(PSLIB_MLPARALLEL_DOWNLOAD_DIR  "${PSLIB_MLPARALLEL_SOURCE_DIR}/src/${PSLIB_MLPARALLEL_PROJECT}")
SET(PSLIB_MLPARALLEL_DST_DIR       "pslib_mlparallel")
SET(PSLIB_MLPARALLEL_INSTALL_ROOT  "${THIRD_PARTY_PATH}/install")
SET(PSLIB_MLPARALLEL_INSTALL_DIR   ${PSLIB_MLPARALLEL_INSTALL_ROOT}/${PSLIB_MLPARALLEL_DST_DIR})
SET(PSLIB_MLPARALLEL_ROOT          ${PSLIB_MLPARALLEL_INSTALL_DIR})
SET(PSLIB_MLPARALLEL_INC_DIR       ${PSLIB_MLPARALLEL_ROOT}/include)
SET(PSLIB_MLPARALLEL_LIB_DIR       ${PSLIB_MLPARALLEL_ROOT}/lib)
SET(PSLIB_MLPARALLEL_LIB           ${PSLIB_MLPARALLEL_LIB_DIR}/libmlparallel_cpu.a)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${PSLIB_MLPARALLEL_ROOT}/lib")

INCLUDE_DIRECTORIES(${PSLIB_MLPARALLEL_INC_DIR})

FILE(WRITE ${PSLIB_MLPARALLEL_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(PSLIB_MLPARALLEL)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${PSLIB_MLPARALLEL_NAME}/include ${PSLIB_MLPARALLEL_NAME}/lib \n"
  "        DESTINATION ${PSLIB_MLPARALLEL_DST_DIR})\n")

ExternalProject_Add(
    ${PSLIB_MLPARALLEL_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${PSLIB_MLPARALLEL_SOURCE_DIR}
    DOWNLOAD_DIR          ${PSLIB_MLPARALLEL_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${PSLIB_MLPARALLEL_URL} -c -q -O ${PSLIB_MLPARALLEL_NAME}.tar.gz
                          && tar zxvf ${PSLIB_MLPARALLEL_NAME}.tar.gz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${PSLIB_MLPARALLEL_INSTALL_ROOT}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${PSLIB_MLPARALLEL_INSTALL_ROOT}
)

ADD_LIBRARY(pslib_mlparallel SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET pslib_mlparallel PROPERTY IMPORTED_LOCATION ${PSLIB_MLPARALLEL_LIB})
ADD_DEPENDENCIES(pslib_mlparallel ${PSLIB_MLPARALLEL_PROJECT})
LIST(APPEND external_project_dependencies pslib_mlparallel)
