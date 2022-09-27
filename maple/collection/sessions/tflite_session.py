import numpy as np
import tflite_runtime.interpreter as tflite


class TFLiteSession:
    def __init__(self, tflite_model_path):
        self.interpreter = tflite.Interpreter(
            model_path=tflite_model_path,
            experimental_delegates=None,
            num_threads=None)

    def onnx_to_tensorrt(self, onnx_model_path):
        success = self.parser.parse_from_file(onnx_model_path)

        if not success:
            for idx in range(self.parser.num_errors):
                print(self.parser.get_error(idx))

            raise ValueError(
                f"Encountered error while converting {onnx_model_path} "
                f"to TensorRT.")

        engine = self.builder.build_engine(self.network, self.config)
        serialized_engine = engine.serialize()

        return serialized_engine

    def setup_inference(self):
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        assert(self.input_details[0]['dtype'] == np.float32)
        # NxHxWxC, H:1, W:2
        # batch_size = self.input_details[0]['shape'][0]
        # height = self.input_details[0]['shape'][1]
        # width = self.input_details[0]['shape'][2]
        # channels = self.input_details[0]['shape'][3]
        # self.input_shape = (batch_size, height, width, channels)
        self.input_shape = self.input_details[0]['shape']
        self.input_data = np.float32(np.random.random(self.input_shape))
        self.interpreter.set_tensor(
            self.input_details[0]['index'], self.input_data)

    def run_inference(self):
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index'])

        return output_data
