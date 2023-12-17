#!/usr/bin/env zsh

command_found () {
  type ${1} &> /dev/null;
}

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

DATA_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/..
MNIST_DIR=${DATA_DIR}/mnist
MNIST_RENDER_DIR=${DATA_DIR}/dataset/mnist_render
rm -rf ${MNIST_RENDER_DIR}
mkdir -p ${MNIST_RENDER_DIR}

${PYTHON} ${MNIST_DIR}/mnist_render.py -out ${MNIST_RENDER_DIR} -num 60000 -seed 101 -dmax 1.0 -dataset -prefix train
${PYTHON} ${MNIST_DIR}/mnist_render.py -out ${MNIST_RENDER_DIR} -num 10000 -seed 102 -dmax 1.0 -dataset -prefix t10k
