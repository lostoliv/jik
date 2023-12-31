#!/usr/bin/env zsh

command_found () {
  type ${1} &> /dev/null;
}

MODE=${1}
if [ -z ${MODE} ] ; then
  MODE="release"
fi

MAKE=make
if ! command_found ${MAKE} ; then
  printf "Command ${MAKE} not found"
  exit 1
fi

CMAKE=cmake
if ! command_found ${CMAKE} ; then
  printf "Command ${CMAKE} not found"
  exit 1
fi

BUILD_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/build
rm -rf ${BUILD_DIR}

if [ ${MODE} = "clean" ] ; then
  exit 0
fi

mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}
if [ ${MODE} = "debug" ] ; then
  ${CMAKE} -DCMAKE_BUILD_TYPE=Debug ..
else
  ${CMAKE} -DCMAKE_BUILD_TYPE=Release ..
fi
${MAKE} -j
popd
