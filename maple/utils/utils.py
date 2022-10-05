"""File contains general utility functions."""
import random
import numpy as np
import os
from pathlib import Path
from maple.conversion import converter_configs as cc
from maple.conversion.converter_configs import NATS_CELL_CONFIG
from maple.utils import nasbench201_utils
from xautodl.models.cell_operations import NAS_BENCH_201


def setup_seed(seed: int, envs: list[str] = []) -> None:
    """Initializes random seed for given environment.

    Args:
        seed: Seed value to use for initialization.
        envs: List of environments to configure seed for ['torch', 'tf'].
    """
    random.seed(seed)
    tseed = random.randint(1, 1E6)
    tcseed = random.randint(1, 1E6)
    npseed = random.randint(1, 1E6)
    ospyseed = random.randint(1, 1E6)

    # PyTorch seeds.
    if 'torch' in envs:
        import torch
        torch.manual_seed(tseed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(tcseed)
            torch.backends.cudnn.deterministic = True

    np.random.seed(npseed)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)

    # Tensorflow seeds.
    if 'tf' in envs:
        import tensorflow as tf
        tf.experimental.numpy.random.seed(npseed)
        tf.random.set_seed(npseed)


def generate_model_uid(arch_idx, input_size):
    """Generates a unique model id for a given model architecture.
    E.g. for a 224x224 input shape and arch_idx 57, the model id is:
    224_57.

    Args:
        arch_idx: Index of the model architecture in NATS-Bench-201.
        input_size: Input dimension of the model (e.g. 224). Only accepts
            a single integer since input shape is expected to be square.
    """
    if input_size is None:
        input_size = cc.DEFAULT_INPUT_SIZE

    return f"{input_size}_{arch_idx}"


def generate_model_outdir(export_dir, dirname):
    return f"{export_dir}/{dirname}"


def generate_model_filename(model_uid, extension, prefix="nats_cell"):
    return f"{prefix}_{model_uid}.{extension}"


def generate_model_out_path(export_dir, dirname, model_uid, extension):
    filename = generate_model_filename(model_uid, extension)
    outdir = generate_model_outdir(export_dir, dirname)
    return f"{outdir}/{filename}"


def generate_model_arch_out_path(export_dir, dirname, model_uid, extension):
    filename = generate_model_filename(model_uid, extension, 
                                       prefix=cc.CELL_FILE_PREFIX)
    outdir = generate_model_outdir(export_dir, dirname)
    return f"{outdir}/{filename}"


def generate_model_ops_out_path(export_dir, dirname, model_uid, extension):
    filename = generate_model_filename(model_uid, extension, 
                                       prefix=cc.OPS_FILE_PREFIX)
    outdir = generate_model_outdir(export_dir, dirname)
    return f"{outdir}/{filename}"


def get_op_key(op_name, w, reduction, outC):
    return nasbench201_utils.op2key(op_name, w//reduction, outC)


def generate_op_keys_dict(width):
    # Generates a dictionary of op_key: configs used to generate op_key
    # {op_key: [op_name, width, reduction, outC]}
    op_keys = dict((get_op_key(op_name, width, reduction, outC),
                    (op_name, width, reduction, outC))
                   for op_name in NAS_BENCH_201
                   for (reduction, inC, outC) in NATS_CELL_CONFIG)

    return op_keys


def generate_backbone_ops_list():
    # Expected stages in backbone model.
    return ['stem', 'resblock1', 'resblock2', 'pool', 'lastact', 'classifier']


def ensure_dir_exists(path_to_dir):
    """Ensures that the given directory exists. If not, creates it."""
    Path(path_to_dir).mkdir(parents=True, exist_ok=True)


def create_outdirs(export_dir: str, model_type: str) -> None:
    """Creates output directories for the given model type.

    Args:
        export_dir: Root directory to export the model to.
        model_type: Type of model to export (onnx, onnx-simplified, tflite).
    """

    cell_dirname = cc.CELL_FILE_CONFIGS[model_type].dirname
    cell_outdir = generate_model_outdir(export_dir, cell_dirname)

    ops_dirname = cc.OPS_FILE_CONFIGS[model_type].dirname
    ops_outdir = generate_model_outdir(export_dir, ops_dirname)

    ensure_dir_exists(cell_outdir)
    ensure_dir_exists(ops_outdir)
