#!/bin/bash

# Usage example:
# ./convert2onnx.sh 0 50 2700
DEFAULT_START="0"
START="${1:-$DEFAULT_START}"

DEFAULT_ITR_PER_BATCH="100"
ITR_PER_BATCH="${2:-$DEFAULT_ITR_PER_BATCH}"

DEFAULT_END="2700"
END="${3:-$DEFAULT_END}"

DEFAULT_INPUT_SIZE="224"
INPUT_SIZE="${4:-$DEFAULT_INPUT_SIZE}"

while true; do
	RANGE_END=$[$START + $ITR_PER_BATCH]
	RANGE_START=$[$START]
	
	if (("$RANGE_END" > "$END")); then
		# Only convert models for the architectures in the specified range.
		break
	fi
	echo "Running convert2onnx.py for range ($RANGE_START, $RANGE_END)"
	python3 convert2onnx.py --range $RANGE_START $RANGE_END --input_size $INPUT_SIZE
	
	START=$[$RANGE_END]
done

echo "Running convert2onnx.py for range ($START, $END)"
python3 convert2onnx.py --range $START $END --input_size $INPUT_SIZE

echo "Model conversion complete."
