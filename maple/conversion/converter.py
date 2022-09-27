from tqdm import tqdm
import glob
import re
import os
from xautodl import models
import multiprocessing as mp
from maple.conversion import converter_configs as conv_cfgs


def filter_generated_models(glob_path, arch_idx_range,
                            regex_arch_uid_match_expr):
    """Parses models already exported to the converted_models_dir output dir and
    filters them from range of arch indices to generate models from.

    This ensures that we don't waste processing time converting models that
    have already been converted (helpful in the event of a previous crash.)

    Returns: list of remaining architecture indices to convert.
    """
    # Generate a set of all architecture indices in target range.
    arch_idxs = set(list(range(*arch_idx_range)))

    # Parse filesystem for filenames of already exported models.
    converted_files = [os.path.basename(x) for x in glob.glob(glob_path)]

    # Regex match expression to parse architecture index from filename.
    r = re.compile(regex_arch_uid_match_expr)
    exported_idxs = set([int(m.group(1)) for m in
                        (r.match(filename) for filename in converted_files)
                        if m])

    # Calculate remaining indices to convert by subtracting exported indices
    # from set of indices to export.
    remaining_idxs = list(arch_idxs.difference(exported_idxs))

    return remaining_idxs


def generate_configs(api, arch_idx_range, out_dir, model_type, input_shape,
                     channels_last):
    """Generates configs for all cell candidate architectures in dataset."""
    export_cfgs = []

    # Get file extension
    file_cfg = conv_cfgs.CELL_FILE_CONFIGS[model_type]
    converted_models_dir = f'{out_dir}/{file_cfg.dirname}'
    converted_models_glob_path = f"{converted_models_dir}/" \
        "*{file_cfg.extension}"

    # Calculate list of remaining architecture indices to convert/export.
    # This may not be the same as the user provided range since some models
    # may have already been exported previously prior to a crashed export
    # session.
    arch_idxs_remaining = filter_generated_models(
        glob_path=converted_models_glob_path,
        arch_idx_range=arch_idx_range,
        regex_arch_uid_match_expr=file_cfg.re_uid_match)

    # Generate export configs for remaining architectures.
    for i in tqdm(arch_idxs_remaining):
        cfg_dict = api.get_net_cfg(i, 'cifar10')

        onnx_cfg = conv_cfgs.ExportConfig(cell_cfg_dict=cfg_dict,
                                          arch_idx=i,
                                          out_dir=out_dir,
                                          input_shape=input_shape,
                                          channels_last=channels_last)

        export_cfgs.append(onnx_cfg)

    print(f"Generated {model_type} configs for range {arch_idx_range} with "
          f"{len(arch_idxs_remaining)} models left to be converted.")

    return export_cfgs


def get_nats_cell(cfg):
    """Returns genotype for NATS cell given config."""
    # Genotype is an object representing the cell structure of a NATS cell,
    # E.g. Structure(4 nodes with |skip_connect~0|+|skip_connect~0|
    # nor_conv_3x3~1|+|none~0|avg_pool_3x3~1|avg_pool_3x3~2|)
    genotype = models.cell_searchs.CellStructure.str2structure(cfg.arch_str)
    return genotype


def to_numpy(tensor):
    """Converts Torch tensor to NumPy."""
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def export_nats_parallel(api, conversion_fn, out_dir, arch_idx_range,
                         input_shape, model_type, channels_last=False):
    """Exports NATS models in parallel using multiprocessing.

    Args:
        api: API object for accessing NATS dataset.
        conversion_fn: Function to convert model to desired format. (e.g.
            convert_nats_to_onnx() or convert_nats_to_tflite()).
        out_dir: Output directory to export models to.
        arch_idx_range: Range of architecture indices to export models from.
        input_shape: Input shape for model.
        model_type: Type of model to export. (e.g. 'onnx' or 'tflite').
        channels_last: Whether to export model with channels last format.
    """
    # Generate configs for model type.
    export_cfgs = generate_configs(api, arch_idx_range, out_dir, model_type,
                                   input_shape=input_shape,
                                   channels_last=channels_last)

    # Use all available cores minus 1 for multiprocessing.
    # Try to leave 1 core free for main process and other CPU processes
    # (e.g. ssh,htop,etc). Note that this isn't guaranteed and in most cases,
    # the scheduler will try to maximize CPU consumption.
    num_cores_to_use = mp.cpu_count()-1
    pool = mp.Pool(num_cores_to_use)

    print(f"Starting conversion process using {num_cores_to_use} cores.")
    pool.map(conversion_fn, export_cfgs)

    print(f"Converted {arch_idx_range[0]}-{arch_idx_range[1]} architectures "
          f"in NATS-Bench to {model_type}. Models exported to {out_dir}")
