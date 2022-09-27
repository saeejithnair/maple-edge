#!/bin/bash

# Usage example:
# ./convert2onnx.sh 0 50 2700
DEFAULT_START="0"
START="${1:-$DEFAULT_START}"

DEFAULT_ITR_PER_BATCH="100"
ITR_PER_BATCH="${2:-$DEFAULT_ITR_PER_BATCH}"

DEFAULT_END="2700"
END="${3:-$DEFAULT_END}"

while true; do
	RANGE_END=$[$START + $ITR_PER_BATCH]
	RANGE_START=$[$START]
	
	if (("$RANGE_END" > "$END")); then
		# Only convert models for the architectures in the specified range.
		break
	fi
	echo "Running convert2onnx.py for range ($RANGE_START, $RANGE_END)"
	python3 convert2onnx.py --range $RANGE_START $RANGE_END
	
	START=$[$RANGE_END]
done

echo "Model conversion complete."
