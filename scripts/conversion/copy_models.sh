#!/bin/bash
set -e 

CHEETAH_MODELS_DIR="/mnt/nas_data/maple-data/models"
LAVAZZA_MODELS_DIR="/pub1/smnair/nas/maple-data/models"

INPUT_SIZE="256"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

INPUT_SIZE="300"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

INPUT_SIZE="32"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

INPUT_SIZE="384"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

INPUT_SIZE="448"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

INPUT_SIZE="512"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

INPUT_SIZE="64"
cd "${CHEETAH_MODELS_DIR}/models_${INPUT_SIZE}"
echo $PWD
rsync -r "vip-lavazza:${LAVAZZA_MODELS_DIR}/models_${INPUT_SIZE}/*" .

