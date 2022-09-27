"""
LatencyRecorder should encapsulate support for recording latency of
multiple types of models and frameworks. A script should be able to
pass in the location of the generated models, and the range of indices
to capture latency for. Upon construction, LatencyRecorder must verify
that all the model files in the range exist at the path provided.

LatencyRecorder must take in an argument specifying the output location
of the pickle file. Regardless of framework or model type, the generated
pickle file must be identical in structure.

LatencyRecorder takes in LatencyRecorderConfig
"""

import numpy as np
import os
from abc import ABC, abstractmethod

from maple.collection.recorder.recorder_configs import LatencyRecorderConfig
from maple.conversion.converter_configs import ExportConfig, FileConfig
from maple.conversion import converter_configs as converter_configs
import maple.utils as utils


class LatencyRecorder(ABC):
    def __init__(self,
                 recorder_config: LatencyRecorderConfig,
                 cell_file_config: FileConfig,
                 ops_file_config: FileConfig) -> None:
        # Engine/framework to use for running inference {'torch', 'keras',
        # 'tflite', 'trt'}.
        self.inference_engine = recorder_config.inference_engine
        # Number of times latency will be recorded for each network.
        self.n_runs = recorder_config.n_runs
        # Number of warmup runs prior to measuring latency.
        self.warmup_runs = 50
        # Device to run inference on {'cpu', 'gpu'}.
        self.device = recorder_config.device
        # List of architecture indices that latency should be recorded for.
        self.target_archs = recorder_config.target_archs

        # Dictionary of {op_key: [op_name, width, reduction, outC]}. Generated
        # based on provided list of op-key strings, or all ops if
        # recorder_config.target_ops == 'all'.
        self.target_ops_dict = self._generate_target_ops_dict(
                                recorder_config.target_ops)

        self.measure_backbone_ops = recorder_config.measure_backbone_ops
        if self.measure_backbone_ops:
            self.backbone_opkey_names = self._generate_target_backbone_ops_list()

        # Path to where latency measurements should be exported to.
        self.output_dir = recorder_config.output_dir
        # Path to where generated models are located.
        self.model_dir = recorder_config.model_dir

        self.cell_file_config = cell_file_config
        self.ops_file_config = ops_file_config

        # Dictionaries mapping model {uid: model_path}
        self.paths_archs, self.paths_ops, self.paths_ops_backbone = self._validate_input_model_files()

    def _validate_input_model_files(self):
        """Validate that the models specified actually exist on the filesystem.
        Raises error if any architecture or ops model cannot be found.

        Returns tuple of 2 dictionaries mapping model uid to model path
        on filesystem. ({arch_idx: arch_model_path}, {op_key: op_model_path})
        """
        missing_archs = []
        paths_archs = {}
        for arch_idx in self.target_archs:
            model_path = utils.generate_model_arch_out_path(
                out_dir=self.model_dir,
                dirname=self.cell_file_config.dirname,
                model_uid=arch_idx,
                extension=self.cell_file_config.extension
            )

            if os.path.isfile(model_path):
                paths_archs[arch_idx] = model_path
            else:
                missing_archs.append(model_path)

        missing_ops = []
        paths_ops = {}
        for op_key in self.target_ops_dict.keys():
            model_path = utils.generate_model_ops_out_path(
                out_dir=self.model_dir,
                dirname=self.ops_file_config.dirname,
                model_uid=op_key,
                extension=self.ops_file_config.extension
            )

            if os.path.isfile(model_path):
                paths_ops[op_key] = model_path
            else:
                missing_ops.append(model_path)

        paths_ops_backbone = None
        if self.measure_backbone_ops:
            paths_ops_backbone = {}
            for backbone_op in self.backbone_opkey_names:
                model_path = utils.generate_model_ops_out_path(
                    out_dir=self.model_dir,
                    dirname=self.ops_file_config.dirname,
                    model_uid=backbone_op,
                    extension=self.ops_file_config.extension
                )

                if os.path.isfile(model_path):
                    paths_ops_backbone[backbone_op] = model_path
                else:
                    missing_ops.append(model_path)

        if len(missing_archs) > 0 or len(missing_ops) > 0:
            raise ValueError(
                f"ERROR, model files missing. Architectures: "
                f"{missing_archs}, OPS: {missing_ops}")

        return (paths_archs, paths_ops, paths_ops_backbone)

    def _generate_target_ops_dict(self, target_ops):
        """Validates list of target ops"""
        # Get dictionary of {op_key: [op_name, width, reduction, outC]}
        supported_ops_dict = utils.generate_op_keys_dict(
            converter_configs.INPUT_SIZE)

        if target_ops == "all":
            # If latency must be measured for all ops, return
            # supported_ops_dict in its entirety.
            return supported_ops_dict

        # If we're measuring latency for only certain ops, copy the
        # key/values for those specific ops from the supported_ops_dict
        # to the target_ops_dict.
        target_ops_dict = {}
        for op_key in target_ops:
            if op_key in supported_ops_dict:
                target_ops_dict[op_key] = supported_ops_dict[op_key]
            else:
                raise ValueError(f"ERROR. Operation {op_key} not supported")

        return target_ops_dict

    def _generate_target_backbone_ops_list(self):
        """Validates list of target ops"""
        # Get list of backbone op key names.
        supported_ops_dict = utils.generate_backbone_ops_list()

        return supported_ops_dict

    @abstractmethod
    def measure_arch_latencies(self):
        pass

    @abstractmethod
    def measure_ops_latencies(self):
        pass

    @abstractmethod
    def _measure_arch_latency(self):
        pass

    @abstractmethod
    def _measure_ops_latency(self):
        pass

    @abstractmethod
    def _latency_on_cpu(self):
        pass

    @abstractmethod
    def _latency_on_gpu(self):
        pass
