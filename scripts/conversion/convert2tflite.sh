#!/bin/bash

# Usage example:
# ./convert2tflite.sh 0 50 2700
DEFAULT_START="0"
START="${1:-$DEFAULT_START}"

DEFAULT_ITR_PER_BATCH="100"
ITR_PER_BATCH="${2:-$DEFAULT_ITR_PER_BATCH}"

DEFAULT_END="2700"
END="${3:-$DEFAULT_END}"

DEFAULT_MODEL_EXPORT_DIR="/home/saeejith/work/nas/maple-data/models"
MODEL_EXPORT_DIR="${4:-$DEFAULT_MODEL_EXPORT_DIR}"

INPUT_SIZES=(16 32 64 128 256 300 384 448 512)

NATS_DIR="/home/${USER}/work/nas/NATS/NATS-tss-v1_0-3ffb9-simple"

while true; do
	RANGE_END=$[$START + $ITR_PER_BATCH]
	RANGE_START=$[$START]
	
	if (("$RANGE_END" > "$END")); then
		# Only convert models for the architectures in the specified range.
		break
	fi

	# Convert models for all input sizes.
	for input_size in ${INPUT_SIZES[@]}; do
		export_dir="${MODEL_EXPORT_DIR}/models_${input_size}"

		echo "Running convert2tflite.py for range ($RANGE_START, $RANGE_END), input: $input_size"
		python3 convert2tflite.py --range $RANGE_START $RANGE_END \
			--input_size $input_size --export_dir $export_dir \
			--nats_dir $NATS_DIR
	done
	START=$[$RANGE_END]
done


# Convert remaining models in specified range.
for input_size in ${INPUT_SIZES[@]}; do
	export_dir="${MODEL_EXPORT_DIR}/models_${input_size}"

	echo "Running convert2tflite.py for range ($START, $END), input: $input_size"
	python3 convert2tflite.py --range $START $END --input_size $input_size
		--export_dir $export_dir --nats_dir $NATS_DIR --convert_ops --convert_backbone
done

echo "Model conversion complete."
