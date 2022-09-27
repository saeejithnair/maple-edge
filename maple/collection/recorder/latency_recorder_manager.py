from maple.collection.recorder.recorder_configs import LatencyRecorderConfig
from latency_recorder import LatencyRecorder
from maple.utils import nasbench201_utils, latency_utils
import utils
from scipy.stats import trim_mean
import numpy as np
from tqdm import tqdm


class LatencyRecorderManager:
    def __init__(self, recorder_config: LatencyRecorderConfig, api=None):
        self.recorder_config = recorder_config
        self.latency_recorder = self._make_recorder(recorder_config)
        self.api = api
        self.dataset = 'cifar10'

        # TODO: this should not be statically hardcoded
        self.input_size = 224

    def _make_recorder(self, recorder_config) -> LatencyRecorder:
        """Creates and returns a Recorder object for either online
        or offline recording mode.
        """
        if recorder_config.online:
            # If models are to be generated on-the-fly.
            return self._make_recorder_online(recorder_config)
        else:
            # If models are to be loaded from exported files.
            return self._make_recorder_offline(recorder_config)

    def _make_recorder_online(self, recorder_config) -> LatencyRecorder:
        """Creates latency recorder for online measurements."""
        pass

    def _make_recorder_offline(self, recorder_config) -> LatencyRecorder:
        """Creates latency recorder for offline measurements."""
        if self.recorder_config.inference_engine == 'trt':
            from latency_recorder_tensorrt import LatencyRecorderTensorRT
            return LatencyRecorderTensorRT(recorder_config)
        elif self.recorder_config.inference_engine == 'tflite':
            from latency_recorder_tflite import LatencyRecorderTFLite
            return LatencyRecorderTFLite(recorder_config)
        elif self.recorder_config.inference_engine == 'torchscript':
            from latency_recorder_torchscript import LatencyRecorderTorchscript
            return LatencyRecorderTorchscript(recorder_config)
        else:
            raise ValueError(
                f"No latency recorder available for inference engine "
                f"{self.recorder_config.inference_engine}")

    def measure_ops_latency(self, enable_perf):
        # Get dict of form {op_key: {'latencies': [], 'perf_stats': }}
        profiling_results = self.latency_recorder.measure_ops_latencies(
                                enable_perf)
        self.ops_latencies = {}
        self.ops_latencies_mean = {}
        self.ops_perf_stats = {}

        for op_key in profiling_results:
            profiling_result = profiling_results[op_key]
            latencies = profiling_result['latencies']

            self.ops_latencies[op_key] = latencies
            self.ops_latencies_mean[op_key] = trim_mean(latencies, 0.1)

            if enable_perf:
                self.ops_perf_stats[op_key] = profiling_result['perf_stats']

        return profiling_results

    def measure_backbone_ops_latency(self):
        # Get dict of form {op_key: {'latencies': [], 'perf_stats': }}
        profiling_results = self.latency_recorder.measure_backbone_ops_latencies()
        self.backbone_ops_latencies = {}
        self.backbone_ops_latencies_mean = {}

        for op_key in profiling_results:
            profiling_result = profiling_results[op_key]
            latencies = profiling_result['latencies']

            self.backbone_ops_latencies[op_key] = latencies
            self.backbone_ops_latencies_mean[op_key] = trim_mean(latencies,
                                                                 0.1)

        return profiling_results

    def get_op_latency(self, ops, C):
        div = {16: 1, 32: 2, 64: 4}
        op_latency = []
        for op in ops:
            # TODO(snair): Validate that this is correct, before used to be
            # w = self.input_size
            input_size = self.input_size / div[C]
            op_key = utils.get_op_key(op_name=op, w=input_size,
                                      reduction=div[C], outC=C)
            op_latency.append(self.ops_latencies_mean[op_key])
        return op_latency

    def get_arch_latency(self, arch_idx):
        latencies = self.arch_latencies[arch_idx]

        return latencies

    def get_backbone_total_latency(self):
        total_latency = 0
        for op_key in self.backbone_ops_latencies_mean:
            total_latency += self.backbone_ops_latencies_mean[op_key]

        return total_latency

    def measure_archs_latency(self):
        # TODO(snair): Why is this returning a value?
        profiling_results_ops = self.measure_ops_latency(enable_perf=False)
        profiling_results_archs = self.latency_recorder.measure_arch_latencies()
        arch_latency_measurements = []
        for idx in tqdm(self.latency_recorder.target_archs):
            config = self.api.get_net_config(idx, self.dataset)
            arch_str = config['arch_str']
            encoding = nasbench201_utils.str2encoding(arch_str)
            # Get a list of op names corresponding to ops in this architecture.
            decoded_arch_str = nasbench201_utils.decode_arch_str(arch_str)
            arch_op_latencies = [
                self.get_op_latency(decoded_arch_str, 16),
                self.get_op_latency(decoded_arch_str, 32),
                self.get_op_latency(decoded_arch_str, 64)
            ]

            profiling_result_arch = profiling_results_archs[idx]
            end2end_arr = profiling_result_arch['latencies']
            end2end = trim_mean(end2end_arr, 0.1)
            # Set summed to NaN since we don't measure stem latency
            summed = np.NaN
            arch_latency_measurements.append(latency_utils.ArchLatencyMeasure(
                idx, end2end, end2end_arr, summed, encoding, arch_str,
                [None, None, None], arch_op_latencies
            ))
            # the [None,None,None] array is simply there as a placeholder.
            # In the future, we'll remove this. It will
            # cause a breaking change right now.

        config = {
            'device': self.recorder_config.device,
            'input_size': self.input_size,
            'online': self.recorder_config.online,
            'inference_engine': self.recorder_config.inference_engine,
            'arch_idx': self.latency_recorder.target_archs,
        }

        # Set to None since we don't collect something similar
        LUT = None

        return (arch_latency_measurements, config, LUT)
