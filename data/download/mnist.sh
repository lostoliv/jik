#!/usr/bin/env zsh

command_found () {
  type ${1} &> /dev/null;
}

CURL=curl
if ! command_found ${CURL}; then
  printf "Command ${CURL} not found"
  exit 1
fi
GUNZIP=gunzip
if ! command_found ${GUNZIP}; then
  printf "Command ${GUNZIP} not found"
  exit 1
fi

MNIST_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/../dataset/mnist
rm -rf ${MNIST_DIR}
mkdir -p ${MNIST_DIR}
pushd ${MNIST_DIR}
${CURL} -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
${CURL} -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
${CURL} -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
${CURL} -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
${GUNZIP} *.gz
popd
