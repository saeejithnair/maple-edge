#!/bin/bash

# Usage example:
# ./convert2onnx.sh 0 50 2700
DEFAULT_START="0"
START="${1:-$DEFAULT_START}"

DEFAULT_ITR_PER_BATCH="100"
ITR_PER_BATCH="${2:-$DEFAULT_ITR_PER_BATCH}"

DEFAULT_END="2700"
END="${3:-$DEFAULT_END}"

DEFAULT_MODEL_EXPORT_DIR="/home/saeejith/work/nas/maple-data/models"
EXPORT_DIR="${DEFAULT_MODEL_EXPORT_DIR}/models_${INPUT_SIZE}"

INPUT_SIZES=(16 32 64 128 256 300 384 448 512)

while true; do
	RANGE_END=$[$START + $ITR_PER_BATCH]
	RANGE_START=$[$START]
	
	if (("$RANGE_END" > "$END")); then
		# Only convert models for the architectures in the specified range.
		break
	fi

	# Convert models for all input sizes.
	for input_size in ${INPUT_SIZES[@]}; do
		export_dir="${DEFAULT_MODEL_EXPORT_DIR}/models_${input_size}"

		echo "Running convert2onnx.py for range ($RANGE_START, $RANGE_END), input: $input_size"
		python3 convert2onnx.py --range $RANGE_START $RANGE_END --input_size $input_size --export_dir $export_dir
	done
	START=$[$RANGE_END]
done


# Convert remaining models in specified range.
for input_size in ${INPUT_SIZES[@]}; do
	export_dir="${DEFAULT_MODEL_EXPORT_DIR}/models_${input_size}"

	echo "Running convert2onnx.py for range ($START, $END), input: $input_size"
	python3 convert2onnx.py --range $START $END --input_size $input_size --export_dir $export_dir --convert_ops --convert_backbone
done

echo "Model conversion complete."
