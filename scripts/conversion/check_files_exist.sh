#!/bin/bash

# This script checks if the files exist in the specified export
# directory by iterating over the specified range.

DEFAULT_START="0"
START="${1:-$DEFAULT_START}"

DEFAULT_END="2700"
END="${2:-$DEFAULT_END}"

DEFAULT_MODEL_TYPE="onnx-simplified"
MODEL_TYPE="${3:-$DEFAULT_MODEL_TYPE}"

DEFAULT_MODEL_EXTENSION="onnx"
MODEL_EXTENSION="${4:-$DEFAULT_MODEL_EXTENSION}"

DEFAULT_MODEL_EXPORT_DIR="/mnt/nas_data/maple-data/models"
MODEL_EXPORT_DIR="${5:-$DEFAULT_MODEL_EXPORT_DIR}"

INPUT_SIZES=(16 32 64 128 256 300 384 448 512)

CUR_ARCH_IDX=$START

while true; do	
	if (("$CUR_ARCH_IDX" > "$END")); then
		# Only check existence of files in the specified range.
		break
	fi

	# Convert models for all input sizes.
	for input_size in ${INPUT_SIZES[@]}; do
		export_dir="${MODEL_EXPORT_DIR}/models_${input_size}/${MODEL_TYPE}/cells"
        file_path="${export_dir}/nats_cell_${input_size}_${CUR_ARCH_IDX}.${MODEL_EXTENSION}"

		if [ ! -f $file_path ]; then
            echo "${file_path} not found!"
        fi
	done
	CUR_ARCH_IDX=$((CUR_ARCH_IDX +1))
done