#!/usr/bin/env zsh

MODE=${1}
if [ -z ${MODE} ] ; then
  MODE="build"
fi

DATA_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/data
rm -rf ${DATA_DIR}/dataset

if [ ${MODE} = "clean" ] ; then
  exit 0
fi

DOWNLOAD_DIR=${DATA_DIR}/download
SHELL=$(ps h -p $$ -o args='' | cut -f1 -d' ')
${SHELL} ${DOWNLOAD_DIR}/mnist.sh
${SHELL} ${DOWNLOAD_DIR}/mnist_render.sh
${SHELL} ${DOWNLOAD_DIR}/svhn.sh
${SHELL} ${DOWNLOAD_DIR}/cifar10.sh
${SHELL} ${DOWNLOAD_DIR}/textgen.sh
