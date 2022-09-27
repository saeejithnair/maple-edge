from dataclasses import dataclass
import numpy as np
from typing import Union, Optional


@dataclass
class LatencyRecorderConfig:
    # Engine/framework to use for running inference
    # {'torch', 'keras', 'tflite', 'tensorrt'}.
    inference_engine: str
    # If online is set to True, model_dir will be ignored and models will
    # be generated on-the-fly. This mode only supports
    # inference_engine={'torch', 'keras'}.
    online: bool
    # Number of times latency will be recorded for each network.
    n_runs: int
    # Input size (assumes square input dimension).
    input_size: int
    # Device to run inference on {'cpu', 'gpu'}.
    device: str
    # List of architecture indices that latency should be recorded for.
    target_archs: np.array
    # List of operations that latency should be recorded for. Each operation
    # should be formatted as an op-key string (e.g. 'avg_pool_3x3_224_16').
    # If set to 'all', will capture latency for all supported OPS.
    target_ops: Union[list, str]
    # Path to where latency measurements should be exported to.
    output_dir: str
    # If backbone operation latencies should be measured.
    measure_backbone_ops: Optional[bool] = False
    # Path to where generated models are located. For online mode,
    # should be set to None.
    model_dir: Optional[str] = None

    def __post_init__(self):
        self._validate_invariants()

    def _validate_invariants(self):
        """Validates config to ensure that all invariants will be preserved."""

        # There are two main to consider: a)online recorder b)offline recorder
        if self.online:
            # Case (a): online recorder (generates models on the fly and
            # measures their latency)
            self._validate_runtime_online()
        else:
            # Case (b): offline recorder (loads generated models from disk)
            self._validate_runtime_offline()

        self._validate_device()

    def _validate_runtime_online(self):
        """Validates configs to ensure that they are appropriate for online
        latency measurement.
        """
        SUPPORTED_ENGINES_ONLINE = ['torch', 'keras']
        if self.inference_engine not in SUPPORTED_ENGINES_ONLINE:
            raise ValueError(
                f"Invalid inference engine '{self.inference_engine}' "
                f"specified for online mode. "
                f"Expected one of {SUPPORTED_ENGINES_ONLINE}")

        if self.model_dir is not None:
            # Model dir should not be specified when running in online mode.
            # Ensure user isn't making a mistake.
            raise ValueError(
                "Model dir has been specified for online mode. "
                "Either set online=False or set model_dir=None.")

    def _validate_runtime_offline(self):
        """Validates configs to ensure that they are appropriate for offline
        latency measurement.
        """
        SUPPORTED_ENGINES_OFFLINE = [
            'torch', 'keras', 'onnx', 'tflite',
            'open-vino', 'trt', 'torchscript']
        if self.inference_engine not in SUPPORTED_ENGINES_OFFLINE:
            raise ValueError(
                f"Invalid inference engine '{self.inference_engine}' "
                " specified for offline mode. "
                f"Expected one of {SUPPORTED_ENGINES_OFFLINE}")

    def _validate_device(self):
        """Validates that specified device is supported."""
        SUPPORTED_DEVICES = ['cpu', 'gpu']
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"Invalid device '{self.device}' specified. "
                             f"Expected one of {SUPPORTED_DEVICES}")
