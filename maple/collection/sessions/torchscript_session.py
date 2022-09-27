import torch


class TorchscriptSession:
    def __init__(self, torchscript_model_path):
        self.net = torch.jit.load(torchscript_model_path)
        self.w = 224
        self.h = 224

    def setup_inference(self, input_shape):
        self.input_data = torch.randn(input_shape).type(torch.float32)

    def run_inference(self):
        output_data = self.net(self.input_data)

        return output_data
