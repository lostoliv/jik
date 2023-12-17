#!/usr/bin/env zsh

command_found () {
  type ${1} &> /dev/null;
}

CURL=curl
if ! command_found ${CURL}; then
  printf "Command ${CURL} not found"
  exit 1
fi
PYTHON=python3
if ! command_found ${PYTHON}; then
  printf "Command ${PYTHON} not found"
  exit 1
fi

PYTHON_NUMPY_VERSION=$(${PYTHON} -c "import numpy ; print(numpy.version.version)")
if [ -z "${PYTHON_NUMPY_VERSION}" ]; then
  printf "Python Numpy module not found - you can install it ('pip install numpy' if you are using pip)"
  exit 1
fi
PYTHON_OPENCV_VERSION=$(${PYTHON} -c "import cv2 ; print(cv2.__version__)")
if [ -z "${PYTHON_OPENCV_VERSION}" ]; then
  printf "Python OpenCV module not found - you can install it ('pip install cv2' if you are using pip)"
  exit 1
fi
PYTHON_SCIPY_VERSION=$(${PYTHON} -c "import scipy ; print(scipy.__version__)")
if [ -z "${PYTHON_SCIPY_VERSION}" ]; then
  printf "Python SciPy module not found - you can install it ('pip install scipy' if you are using pip)"
  exit 1
fi

DATA_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/..
MNIST_DIR=${DATA_DIR}/mnist
SVHN_DIR=${DATA_DIR}/dataset/svhn
rm -rf ${SVHN_DIR}
mkdir -p ${SVHN_DIR}
pushd ${SVHN_DIR}

URL=http://ufldl.stanford.edu/housenumbers

TEST_FILE=test_32x32.mat
${CURL} -O ${URL}/${TEST_FILE}
${PYTHON} ${MNIST_DIR}/svhn_to_mnist.py   -svhn ${SVHN_DIR}/${TEST_FILE} -out ${SVHN_DIR} -prefix t10k
${PYTHON} ${MNIST_DIR}/svhn_to_cifar10.py -svhn ${SVHN_DIR}/${TEST_FILE} -out ${SVHN_DIR} -prefix test
rm -f ${TEST_FILE}

TRAIN_FILE=train_32x32.mat
${CURL} -O ${URL}/${TRAIN_FILE}
${PYTHON} ${MNIST_DIR}/svhn_to_mnist.py   -svhn ${SVHN_DIR}/${TRAIN_FILE} -out ${SVHN_DIR} -prefix train
${PYTHON} ${MNIST_DIR}/svhn_to_cifar10.py -svhn ${SVHN_DIR}/${TRAIN_FILE} -out ${SVHN_DIR} -prefix data
rm -f ${TRAIN_FILE}

popd
