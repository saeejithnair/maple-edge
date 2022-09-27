"""File contains general utility functions."""
import random
import numpy as np
import os
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


def generate_model_filename(model_uid, extension, prefix="nats_cell"):
    return f"{prefix}_{model_uid}.{extension}"


def generate_model_out_path(out_dir, dirname, model_uid, extension):
    filename = generate_model_filename(model_uid, extension)
    return f"{out_dir}/{dirname}/{filename}"


def generate_model_arch_out_path(out_dir, dirname, model_uid, extension):
    filename = generate_model_filename(model_uid, extension, 
                                       prefix=cc.CELL_FILE_PREFIX)
    return f"{out_dir}/{dirname}/{filename}"


def generate_model_ops_out_path(out_dir, dirname, model_uid, extension):
    filename = generate_model_filename(model_uid, extension, 
                                       prefix=cc.OPS_FILE_PREFIX)
    return f"{out_dir}/{dirname}/{filename}"


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
