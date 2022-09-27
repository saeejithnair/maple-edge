import numpy as np
import time
from tqdm import tqdm

from latency_recorder import LatencyRecorder
from maple.collection.sessions.torchscript_session import TorchscriptSession
from maple.conversion.converter_configs import CELL_FILE_CONFIGS
from maple.conversion.converter_configs import OPS_FILE_CONFIGS
from maple.conversion.converter_configs import TORCHSCRIPT_OPS_INPUT_SHAPE
from perf_recorder import PerfRecorder
from recorder_configs import LatencyRecorderConfig


class LatencyRecorderTorchscript(LatencyRecorder):
    def __init__(self, recorder_config: LatencyRecorderConfig) -> None:
        super().__init__(
            recorder_config,
            cell_file_config=CELL_FILE_CONFIGS['torchscript'],
            ops_file_config=OPS_FILE_CONFIGS['torchscript'])

        if self.device != 'cpu':
            raise ValueError(
                f"ERROR, received device {self.device}. "
                f"Only CPU based inference is supported for Torchscript.")

        self.arch_input_shape = (1, 3, 224, 224)

    def _latency_on_cpu(self,
                        torchscript_session: TorchscriptSession,
                        enable_perf=False):
        profiling_result = {
            'latencies': np.zeros((self.n_runs, 1)),
            'perf_stats': None
        }

        for ii in range(self.warmup_runs):
            _ = torchscript_session.run_inference()

        if enable_perf:
            perf_recorder = PerfRecorder()
            perf_recorder.start_profiling()

        for ii in range(self.n_runs):
            start = time.time()
            _ = torchscript_session.run_inference()
            end = time.time()
            profiling_result['latencies'][ii] = end-start

        if enable_perf:
            perf_recorder.stop_profiling()
            profiling_result['perf_stats'] = perf_recorder.get_stats()

        return profiling_result

    def _latency_on_gpu(self):
        raise NotImplementedError(
            "GPU based inference is not supported for Torchscript.")

    def _measure_arch_latency(self, ort_session, ort_inputs):
        raise NotImplementedError("Not implemented.")

    def _measure_ops_latency(self, ort_session, ort_inputs):
        raise NotImplementedError("Not implemented.")

    def measure_arch_latencies(self):
        profiling_results = {}
        print(f"Measuring latencies for {len(self.paths_archs.items())} "
              f"architectures.")
        for arch_idx, model_path in tqdm(self.paths_archs.items()):
            torchscript_session = TorchscriptSession(
                                    torchscript_model_path=model_path)

            torchscript_session.setup_inference(self.arch_input_shape)
            profiling_result = self._latency_on_cpu(
                                    torchscript_session, enable_perf=False)
            profiling_results[arch_idx] = profiling_result

        return profiling_results

    def measure_ops_latencies(self, enable_perf):
        profiling_results = {}
        for op_key, model_path in self.paths_ops.items():
            torchscript_session = TorchscriptSession(
                                    torchscript_model_path=model_path)

            input_shape = TORCHSCRIPT_OPS_INPUT_SHAPE[op_key]
            torchscript_session.setup_inference(input_shape)
            profiling_result = self._latency_on_cpu(
                                torchscript_session, enable_perf)

            profiling_results[op_key] = profiling_result

        return profiling_results

    def measure_backbone_ops_latencies(self):
        profiling_results = {}
        for op_key, model_path in tqdm(self.paths_ops_backbone.items()):
            torchscript_session = TorchscriptSession(
                                    torchscript_model_path=model_path)

            input_shape = TORCHSCRIPT_OPS_INPUT_SHAPE[op_key]
            torchscript_session.setup_inference(input_shape)
            profiling_result = self._latency_on_cpu(
                                torchscript_session, enable_perf=False)
            profiling_results[op_key] = profiling_result

        return profiling_results
