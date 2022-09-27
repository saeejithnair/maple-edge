import numpy as np
# When imported, automatically performs all the steps necessary
# to get CUDA ready for submission of compute kernels.
# https://documen.tician.de/pycuda/util.html
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from maple.collection.recorder.perf_recorder import PerfRecorder


class TRTSession:
    def __init__(self, enable_perf=False):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.explicit_batch_flag = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.enable_perf = enable_perf

    def build_engine_from_onnx(self, onnx_model_path):
        with (self.logger, trt.Builder(self.logger) as builder,
              builder.create_network(self.explicit_batch_flag) as network,
              trt.OnnxParser(network, self.logger) as parser,
              builder.create_builder_config() as config):
            success = parser.parse_from_file(onnx_model_path)
            if not success:
                for idx in range(parser.num_errors):
                    print(parser.get_error(idx))

                raise ValueError(
                    f"Encountered error while converting "
                    f"{onnx_model_path} to TensorRT.")

            config.max_workspace_size = 1 << 20  # 1 MiB
            engine = builder.build_engine(network, config)

        return engine

    def onnx_to_tensorrt(self, onnx_model_path):
        with self.build_engine_from_onnx(onnx_model_path) as engine:
            serialized_engine = engine.serialize()

        return serialized_engine

    def deserialize_engine(self, serialized_engine):
        with trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)

        return engine

    def build_engine_from_serialized_file(self, trt_model_path):
        with open(trt_model_path, "rb") as f:
            serialized_engine = f.read()

        engine = self.deserialize_engine(serialized_engine)
        return engine

    def profile_latency(self, context, bindings, stream,
                        n_runs, n_warmup_runs):
        profiling_results = {
            'latencies': np.zeros((n_runs, 1)),
            'perf_stats': None
        }

        start = cuda.Event()
        end = cuda.Event()

        # Warmup
        for _ in range(n_warmup_runs):
            context.execute_async_v2(bindings=bindings,
                                     stream_handle=stream.handle)
            stream.synchronize()

        # If perf metrics should be collected, start profiling.
        if self.enable_perf:
            perf_recorder = PerfRecorder()
            perf_recorder.start_profiling()

        for ii in range(n_runs):
            # Start latency measurement.
            start.record(stream)

            # Run inference.
            context.execute_async_v2(bindings=bindings,
                                     stream_handle=stream.handle)
            # Stop latency measurement.
            end.record(stream)
            stream.synchronize()

            # Convert from milliseconds to seconds and store in array.
            profiling_results['latencies'][ii] = end.time_since(start)/1000

        if self.enable_perf:
            perf_recorder.stop_profiling()
            profiling_results['perf_stats'] = perf_recorder.get_stats()

        return profiling_results

    def measure_inference_latency(self, serialized_model_path, n_runs,
                                  n_warmup_runs):
        with (open(serialized_model_path, "rb") as f,
              trt.Runtime(self.logger) as runtime,
              runtime.deserialize_cuda_engine(f.read()) as engine,
              engine.create_execution_context() as context):
            # Determine dimensions and create page-locked memory buffers
            # (i.e. won't be swapped to disk) to hold host inputs/outputs.

            # Allocate host input memory.
            h_input = cuda.pagelocked_empty(trt.volume(
                        engine.get_binding_shape(0)), dtype=np.float32)
            h_input[:] = np.random.random(h_input.shape)
            # Allocate device input memory.
            d_input = cuda.mem_alloc(h_input.nbytes)

            # Create bindings for buffers
            bindings = [int(d_input)]

            # Setup output tensors
            host_output_tensors = []
            device_output_tensors = []

            # Assumes that there is only 1 input tensor.
            num_output_tensors = engine.num_bindings - 1
            for i in range(num_output_tensors):
                # Offset by 1 to skip index for input binding
                binding_idx = i+1

                # Allocate host output memory.
                h_output = cuda.pagelocked_empty(trt.volume(
                            engine.get_binding_shape(binding_idx)),
                            dtype=np.float32)
                h_output[:] = np.random.random(h_output.shape)
                host_output_tensors.append(h_output)

                # Allocate device output memory.
                d_output = cuda.mem_alloc(h_output.nbytes)
                device_output_tensors.append(d_output)
                bindings.append(int(d_output))

            # Create stream in which to copy inputs/outputs and run inference.
            stream = cuda.Stream()

            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, h_input, stream)

            # Get profiling results {latencies, perf_stats (if enabled)}.
            profiling_results = self.profile_latency(
                context, bindings, stream, n_runs, n_warmup_runs)

            # Copy outputs back to host so we can clean up memory.
            for ii in range(num_output_tensors):
                h_output = host_output_tensors[ii]
                d_output = device_output_tensors[ii]
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                d_output.free()

            d_input.free()
            stream.synchronize()

        return profiling_results
