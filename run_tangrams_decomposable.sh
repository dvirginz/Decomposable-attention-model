#!/usr/bin/env bash
# Run this script from the root of the project

BASE_DIR=$(cd $(dirname $0);pwd -P)
export MKL_THREADING_LAYER="GNU"
export LD_LIBRARY_PATH="/usr/local/lib/cuda-8.0/lib64/:/usr/local/lib/cudnn-8.0-v7/lib64/"
export STRONGSUP_DIR=${BASE_DIR}/data

echo "=================== Start running script ==================="
if [ -z "$1" ]
    then echo "No GPU supplied, running on CPU"
else
    echo "Using GPU $1"
    export CUDA_VISIBLE_DEVICES=$1
fi

PYTHONPATH=${BASE_DIR}:${BASE_DIR}/third-party/gtd/ python ${BASE_DIR}/scripts/main-decomposable.py ${BASE_DIR}/configs/rlong/best-tangrams.txt
echo "=================== Finish running script ==================="
