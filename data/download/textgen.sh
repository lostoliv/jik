#!/usr/bin/env zsh

command_found () {
  type ${1} &> /dev/null;
}

CURL=curl
if ! command_found ${CURL}; then
  printf "Command ${CURL} not found"
  exit 1
fi

TEXTGEN_DIR=$(readlink -f $(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd))/../dataset/textgen
rm -rf ${TEXTGEN_DIR}
mkdir -p ${TEXTGEN_DIR}
pushd ${TEXTGEN_DIR}
${CURL} -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
popd
