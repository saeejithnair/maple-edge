import argparse
import os
import numpy as np
from nats_bench import create
import onnx
import onnxruntime
from onnxsim import simplify
import sys
import torch
import torch.nn as nn
import xautodl
from xautodl import config_utils
from xautodl.models.cell_infers import TinyNetwork as TinyNetworkTorch
from xautodl.models.cell_operations import NAS_BENCH_201
from xautodl.models.cell_operations import OPS as OPS_TORCH

from maple.conversion import converter
from maple.conversion import converter_configs as conv_cfgs
from maple.utils import utils

"""
ONNX conversion script for NAS-Bench-201.

This script asynchronously converts the PyTorch networks in the NAS-Bench-201
dataset to a simplified ONNX format. The ONNX models are exported to the
specified output directory. The script also validates that the output of
the ONNX networks matches the output of the PyTorch networks.

Pipeline execution sequence works as follows:
1. Load NATS-Bench-201 dataset/create NATS-Bench-201 API
2. For each cell architecture in the dataset:
    a. Convert NATS cell to PyTorch network
    b. Export PyTorch network to ONNX format
    c. Validate that the output of the ONNX model matches the output of the
        PyTorch network
    d. Save ONNX model to disk
    e. Simplify ONNX model
    f. Validate that the output of the simplified ONNX model matches the output
        of the PyTorch network
    g. Save simplified ONNX model to disk
3. Convert NATS-Bench-201 operations to ONNX-simplified format
4. Convert NATS-Bench-201 backbone to ONNX-simplified format
"""


def validate_onnx_model(torch_net, onnx_model_path, dummy_input):
    """Validate that the output of the ONNX network matches output of PyTorch
    network. Raises AssertionError if output does not match
    within specified tolerance.

    Args:
        torch_net: PyTorch network
        onnx_model_path: Path to ONNX model
        dummy_input: Dummy input to use for validation
    """
    # Get output from PyTorch network.
    torch_net_output = torch_net(dummy_input)
    if len(torch_net_output) == 2:
        weights, torch_net_output = torch_net_output

    torch_net_output = converter.to_numpy(torch_net_output)

    # Get output from ONNX network.
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name:
                  converter.to_numpy(dummy_input)}
    onnx_net_output = ort_session.run(None, ort_inputs)

    if len(onnx_net_output) == 2:
        weights, onnx_net_output = onnx_net_output

    if not np.allclose(torch_net_output, onnx_net_output,
                       rtol=1e-02, atol=1e-05):
        raise ValueError(f"Validation failed. Output for ONNX model "
                         f"{onnx_model_path} does not match output "
                         "of PyTorch network.")


def simplify_onnx_model(onnx_model_path, simplified_model_output_path):
    """Loads an existing ONNX model, simplifies it using onnx-simplifier,
    and then exports the model to the specified model output path.

    Args:
        onnx_model_path: Path to ONNX model to simplify
        simplified_model_output_path: Path to export simplified ONNX model

    Raises assertion error if the simplified model is not valid.
    """
    # Load ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Convert model to simplified form
    onnx_model_simplified, check = simplify(onnx_model)

    if not check:
        raise ValueError(f"Failed to simplify ONNX model {onnx_model_path}.")

    onnx.save(onnx_model_simplified, simplified_model_output_path)


def convert_torch_to_onnx(net, model_out_path, simplified_model_out_path,
                          dummy_input):
    """Converts a PyTorch network to ONNX and then simplifies the model.

    Args:
        net: PyTorch network
        model_out_path: Path to export ONNX model
        simplified_model_out_path: Path to export simplified ONNX model
        dummy_input: Dummy input to use for validation

    Raises assertion error if any step in the conversion process fails.
    """
    # Set network to eval mode prior to exporting
    net.eval()

    # Export PyTorch network to ONNX format and save to disk.
    torch.onnx.export(net, dummy_input, model_out_path, input_names=['input'],
                      output_names=['output'])

    # Validate that export happened correctly.
    validate_onnx_model(torch_net=net, onnx_model_path=model_out_path,
                        dummy_input=dummy_input)
    print(f'Exported and validated ONNX model {model_out_path}')
    sys.stdout.flush()

    # Simplify ONNX model
    simplify_onnx_model(onnx_model_path=model_out_path,
                        simplified_model_output_path=simplified_model_out_path)

    # Validate that the simplified model is still correct.
    validate_onnx_model(torch_net=net,
                        onnx_model_path=simplified_model_out_path,
                        dummy_input=dummy_input)

    print(f"Exported and validated simplified ONNX model "
          f"{simplified_model_out_path}")
    sys.stdout.flush()


def convert_nats_ops_to_onnx(out_dir, input_shape):
    """Converts the NATS-Bench-201 operations to ONNX-simplified and
    saves them to disk.

    Args:
        out_dir: Directory to save ONNX models to
        input_shape: Shape of input to use for validation
    """
    print('Converting NATS ops to onnx-simplified')
    # Reduction, inC (input channels) and outC (output channels)
    cell_config = [(1, 16, 16), (2, 32, 32), (4, 64, 64)]
    w, h = input_shape
    for op_name in NAS_BENCH_201:
        for (reduction, inC, outC) in cell_config:
            input_v = torch.randn(1, inC, w//reduction, h//reduction)

            # Assume op depth = 1
            opnet = get_ops_network(op_name, inC, outC)
            op_key = utils.get_op_key(op_name, w, reduction, outC)
            file_cfg_onnx = conv_cfgs.OPS_FILE_CONFIGS['onnx']
            model_out_path = utils.generate_model_ops_out_path(
                out_dir=out_dir,
                dirname=file_cfg_onnx.dirname,
                model_uid=op_key,
                extension=file_cfg_onnx.extension)

            file_cfg_onnx_simp = conv_cfgs.OPS_FILE_CONFIGS['onnx-simplified']
            simplified_model_out_path = utils.generate_model_ops_out_path(
                out_dir=out_dir,
                dirname=file_cfg_onnx_simp.dirname,
                model_uid=op_key,
                extension=file_cfg_onnx_simp.extension)

            convert_torch_to_onnx(opnet, model_out_path,
                                  simplified_model_out_path,
                                  input_v)


class Backbone():
    """Class to represent the backbone/input stem of a cell.

    Args:
        inC: Number of input channels
        n_classes: Number of classes in the dataset
        h: Height of input
        w: Width of input
    """
    def __init__(self, inC=16, n_classes=10, h=224, w=224) -> None:
        self.inC = inC
        self.n_classes = n_classes
        self.h = h
        self.w = w
        self.torch_device = 'cpu'

        self._setup_static_blocks()
        self._setup_static_inputs(self.h, self.w)

    def _setup_static_blocks(self):
        """Sets up the static blocks of the backbone. These are common across
        all cells in NATS-Bench-201.

        Args:
            None
        """
        self.blocks = {
            'stem':
                nn.Sequential(
                    nn.Conv2d(3, self.inC, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.inC)),
            'resblock1':
                xautodl.models.cell_operations.ResNetBasicblock(
                    self.inC, self.inC*2, 2, True),
            'resblock2':
                xautodl.models.cell_operations.ResNetBasicblock(
                    self.inC*2, self.inC*4, 2, True),
            'pool': nn.AdaptiveAvgPool2d((1, 1)),
            'lastact':
                nn.Sequential(
                    nn.BatchNorm2d(self.inC*4), nn.ReLU(inplace=True)),
            'classifier': nn.Linear(self.inC*4, self.n_classes),
        }

    def _setup_static_inputs(self, h, w):
        """Sets up the static inputs of the backbone. These are fixed for
        each static block, and only depend on the input shape.

        Args:
            h: Height of input
            w: Width of input
        """
        self.inputs = {
            'stem': torch.randn(
                1, 3, w, h, device=self.torch_device, dtype=torch.float32),
            'resblock1': torch.randn(
                1, self.inC, w, h, device=self.torch_device,
                dtype=torch.float32),
            'resblock2': torch.randn(
                1, self.inC*2, w//2, h//2, device=self.torch_device,
                dtype=torch.float32),
            'pool': torch.randn(
                1, self.inC*4, w//4, h//4, device=self.torch_device,
                dtype=torch.float32),
            'lastact': torch.randn(
                1, self.inC*4, w//4, h//4, device=self.torch_device,
                dtype=torch.float32),
            'classifier': torch.randn(
                1, self.inC*4, device=self.torch_device, dtype=torch.float32)}

    def get_block_names(self):
        """Returns the names of the static blocks in the backbone."""
        return list(self.blocks.keys())

    def get_block(self, block_name):
        """Returns the static block with the given name."""
        return self.blocks[block_name]

    def get_input(self, block_name):
        """Returns the static input to the given static block."""
        return self.inputs[block_name]


def convert_nats_backbone_to_onnx(out_dir, input_shape):
    """Converts the NATS-Bench-201 backbone to ONNX-simplified and
    saves it to disk.

    Args:
        out_dir: Directory to save ONNX models to
        input_shape: Shape of input to use for validation
    """
    inC = 16
    n_classes = 10
    w, h = input_shape
    print("Converting NATS backbone to ONNX.")

    backbone = Backbone(inC, n_classes, w=w, h=h)

    # Convert each layer in backbone to ONNX.
    backbone_blocks = backbone.get_block_names()
    for block_name in backbone_blocks:
        backbone_block_net = backbone.get_block(block_name)
        backbone_block_input = backbone.get_input(block_name)

        # Generate path for saving ONNX model.
        file_cfg_onnx = conv_cfgs.OPS_FILE_CONFIGS['onnx']
        model_out_path = utils.generate_model_ops_out_path(
            out_dir=out_dir,
            dirname=file_cfg_onnx.dirname,
            model_uid=block_name,
            extension=file_cfg_onnx.extension)

        # Generate path for saving simplified ONNX model.
        file_cfg_onnx_simp = conv_cfgs.OPS_FILE_CONFIGS['onnx-simplified']
        simplified_model_out_path = utils.generate_model_ops_out_path(
            out_dir=out_dir,
            dirname=file_cfg_onnx_simp.dirname,
            model_uid=block_name,
            extension=file_cfg_onnx_simp.extension)

        # Convert to ONNX and save.
        convert_torch_to_onnx(backbone_block_net, model_out_path,
                              simplified_model_out_path,
                              backbone_block_input)


def get_torch_network(config, channels_last=True):
    """Returns the PyTorch network given a NATS cell architecture index."""
    genotype = converter.get_nats_cell(config)

    # Convert NATS cell to PyTorch network
    torch_net = TinyNetworkTorch(config.C, config.N, genotype,
                                 config.num_classes)

    if channels_last:
        torch_net = torch_net.to(memory_format=torch.channels_last)

    return torch_net


def get_ops_network(op_name, C_in, C_out):
    """Returns the PyTorch network given a NATS cell op name.

    Args:
        op_name: Name of op to use
        C_in: Number of input channels
        C_out: Number of output channels
    """
    return OPS_TORCH[op_name](C_in, C_out, 1, True, True)


def convert_nats_to_onnx(export_config: conv_cfgs.ExportConfig):
    """Converts cells in NATS-Bench-201 to ONNX and saves them to disk.

    Args:
        export_config: Stores info about cell to be converted.
    """
    cell_config_dict = export_config.cell_config_dict
    # Converts cell config from a dict to a namedtuple.
    cell_config = config_utils.dict2config(cell_config_dict, None)
    w, h = export_config.input_shape
    dummy_input = torch.randn(1, 3, w, h).type(torch.float32)

    if export_config.channels_last:
        dummy_input = dummy_input.to(memory_format=torch.channels_last)

    net = get_torch_network(cell_config, export_config.channels_last)

    # Generate path for saving ONNX model.
    file_cfg_onnx = conv_cfgs.CELL_FILE_CONFIGS['onnx']
    model_out_path = utils.generate_model_out_path(
                        out_dir=export_config.out_dir,
                        dirname=file_cfg_onnx.dirname,
                        model_uid=export_config.arch_idx,
                        extension=file_cfg_onnx.extension)

    # Generate path for saving simplified ONNX model.
    file_cfg_onnx_simp = conv_cfgs.CELL_FILE_CONFIGS['onnx-simplified']
    simplified_model_out_path = utils.generate_model_out_path(
                                    out_dir=export_config.out_dir,
                                    dirname=file_cfg_onnx_simp.dirname,
                                    model_uid=export_config.arch_idx,
                                    extension=file_cfg_onnx_simp.extension)

    # Convert to ONNX and save.
    try:
        convert_torch_to_onnx(net, model_out_path,
                              simplified_model_out_path, dummy_input)
    except Exception as e:
        print(f"ERROR. Encountered exception while trying to convert "
              f"architecture {export_config.arch_idx}")
        print(e)
        sys.stdout.flush()
        # An error may have been raised in another worker process while this
        # process was in the middle of exporting a model. We don't want
        # corrupt files so delete any model that may not have been
        # successfully exported and validated.
        if os.path.exists(simplified_model_out_path):
            os.remove(simplified_model_out_path)
        if os.path.exists(model_out_path):
            os.remove(model_out_path)


def convert_nats_to_onnx_parallel(api, out_dir, arch_idx_range, input_shape,
                                  channels_last=False):
    """Convert NATS network from Pytorch to Simplified ONNX in parallel using
    multiprocessing.

    Args:
        api: NATS-Bench-201 API
        out_dir: Directory to save ONNX models to
        arch_idx_range: Range of architecture indices to convert
        input_shape: Shape of input to use for validation
        channels_last: Whether to use channels last memory format
    """
    converter.export_nats_parallel(api, convert_nats_to_onnx, out_dir,
                                   arch_idx_range, input_shape,
                                   model_type='onnx-simplified',
                                   channels_last=channels_last)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=27, dest="seed",
        help="Seed value for random generator.")
    parser.add_argument(
        "--input_shape", default=conv_cfgs.INPUT_SIZE,
        dest="input_shape", help="Input size for models.")
    parser.add_argument(
        "--export_dir",
        default="/home/saeejith/work/nas/maple-data/models",
        dest="export_dir",
        help="Output directory to store exported ONNX files.")
    parser.add_argument(
        "--nats_dir",
        default="/home/saeejith/work/nas/NATS/NATS-tss-v1_0-3ffb9-simple",
        dest="nats_dir",
        help="Path to directory containing NATS-tss-v1_0-3ffb9-simple dataset")
    parser.add_argument(
        "--range", nargs=2, default=[0, 15625], dest="range", type=(int),
        help="Range of architectures to export")
    parser.add_argument(
        '--convert_ops', dest='convert_ops', action='store_true')
    parser.add_argument(
        '--convert_backbone', dest='convert_backbone', action='store_true')

    args = parser.parse_args()
    assert(len(args.range) == 2)

    # TODO(snair): Make input shape dynamic.
    if args.input_shape != conv_cfgs.INPUT_SIZE:
        # Input shape must be fixed to this for compatibility with previously
        # collected data.
        raise ValueError(f"Input shape must be "
                         f"{(conv_cfgs.INPUT_SIZE,conv_cfgs.INPUT_SIZE)}. "
                         f"Received ({args.input_shape},{args.input_shape})")

    input_shape = (args.input_shape, args.input_shape)

    utils.setup_seed(args.seed, envs=['torch'])

    # Create NATS-Bench API
    api = create(args.nats_dir, 'tss', fast_mode=True, verbose=False)

    # Execute conversion.
    convert_nats_to_onnx_parallel(api, args.export_dir, args.range,
                                  input_shape)

    if args.convert_ops:
        convert_nats_ops_to_onnx(args.export_dir, input_shape)

    if args.convert_backbone:
        convert_nats_backbone_to_onnx(args.export_dir, input_shape)
