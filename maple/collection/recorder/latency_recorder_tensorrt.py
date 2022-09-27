import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from tqdm import tqdm

from latency_recorder import LatencyRecorder
from recorder_configs import LatencyRecorderConfig
from maple.collection.sessions.tensorrt_session import TRTSession
from maple.conversion.converter_configs import CELL_FILE_CONFIGS
from maple.conversion.converter_configs import OPS_FILE_CONFIGS


class LatencyRecorderTensorRT(LatencyRecorder):
    def __init__(self, recorder_config: LatencyRecorderConfig) -> None:
        super().__init__(recorder_config,
                         cell_file_config=CELL_FILE_CONFIGS['trt'],
                         ops_file_config=OPS_FILE_CONFIGS['trt'])

        if self.device != 'gpu':
            raise ValueError(
                f"ERROR, received device {self.device}. "
                f"Only GPU based inference is supported for TensorRT.")

    def _latency_on_cpu(self):
        raise NotImplementedError(
            "CPU based inference is not supported for TensorRT.")

    def _latency_on_gpu(self, trt_session: TRTSession,
                        serialized_model_path: str):
        profiling_result = trt_session.measure_inference_latency(
                                serialized_model_path,
                                self.n_runs, self.warmup_runs)

        return profiling_result

    def _measure_arch_latency(self, ort_session, ort_inputs):
        raise NotImplementedError("Not implemented.")

    def _measure_ops_latency(self, ort_session, ort_inputs):
        raise NotImplementedError("Not implemented.")

    def measure_arch_latencies(self):
        profiling_results = {}
        for arch_idx, model_path in tqdm(self.paths_archs.items()):
            trt_session = TRTSession(enable_perf=False)
            profiling_result = self._latency_on_gpu(trt_session, model_path)
            profiling_results[arch_idx] = profiling_result

        return profiling_results

    def measure_ops_latencies(self, enable_perf):
        profiling_results = {}
        for op_key, model_path in self.paths_ops.items():
            trt_session = TRTSession(enable_perf)
            profiling_result = self._latency_on_gpu(trt_session, model_path)
            profiling_results[op_key] = profiling_result

        return profiling_results

    def measure_backbone_ops_latencies(self):
        profiling_results = {}
        for op_key, model_path in tqdm(self.paths_ops_backbone.items()):
            trt_session = TRTSession(enable_perf=False)
            profiling_result = self._latency_on_gpu(trt_session, model_path)
            profiling_results[op_key] = profiling_result

        return profiling_results
