#!/bin/bash

export CONTAINER_NAME="env_cu118_ncu_cll"
export IMAGE_NAME="pytorch251_cuda118_cudnn9_py311_ssh_cll:v1.0"
export LOCAL_DIR="/data3/cll/codes"
export TGT_DIR="/data3/cll/codes"

docker run -itd \
--rm \
--gpus all \
--shm-size=16G \
--cap-add=SYS_ADMIN \
-v "${LOCAL_DIR}:${TGT_DIR}" \
--name "${CONTAINER_NAME}" \
"${IMAGE_NAME}"
