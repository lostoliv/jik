#!/usr/bin/env zsh

command_found () {
  type ${1} &> /dev/null;
}

CURL=curl
if ! command_found ${CURL}; then
  printf "Command ${CURL} not found"
  exit 1
fi
TAR=tar
if ! command_found ${TAR}; then
  printf "Command ${TAR} not found"
  exit 1
fi

CIFAR10_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/../dataset/cifar10
rm -rf ${CIFAR10_DIR}
mkdir -p ${CIFAR10_DIR}
pushd ${CIFAR10_DIR}
${CURL} -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
${TAR} xzf *.tar.gz --strip-components=1
rm -f *.tar.gz
popd
