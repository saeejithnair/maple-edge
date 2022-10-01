
import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tensorrt as trt
import tqdm

from maple.utils import utils
from tensorrt_session import TRTSession

ONNX_EXTENSION = 'onnx'
TRT_EXTENSION = 'engine'

def get_files_from_dir(dir, extension):
    files_list = glob.glob(f'{dir}/*.{extension}')

    return files_list


def convert_to_tensorrt(export_configs):
    onnx_filename, onnx_models_dir, trt_output_dir = export_configs

    try:
        onnx_file_path = f"{onnx_models_dir}/{onnx_filename}.{ONNX_EXTENSION}"
        trt_output_path = f"{trt_output_dir}/{onnx_filename}.{TRT_EXTENSION}"
        
        trt_session = TRTSession()
        serialized_engine = trt_session.onnx_to_tensorrt(onnx_file_path)
        with open(trt_output_path, "wb") as f:
            print(f"Started writing {trt_output_path} to disk.")
            sys.stdout.flush()
            f.write(serialized_engine)
            print(f"Finished writing {trt_output_path} to disk.")
            sys.stdout.flush()
    except Exception as e:
        print(f'ERROR. Encountered exception while trying to convert architecture {onnx_filename}')
        print(e)
        if os.path.exists(trt_output_path):
            os.remove(trt_output_path)
            print(f"Removed TRT file {trt_output_path}.")

def filter_generated_models(onnx_models_dir, trt_output_dir):    
    onnx_models_dir = os.path.abspath(onnx_models_dir)
    trt_output_dir = os.path.abspath(trt_output_dir)

    onnx_file_paths = get_files_from_dir(onnx_models_dir, extension=ONNX_EXTENSION)
    trt_file_paths = get_files_from_dir(trt_output_dir, extension=TRT_EXTENSION)

    onnx_filenames = set([Path(file_path).stem for file_path in onnx_file_paths])
    trt_filenames = set([Path(file_path).stem for file_path in trt_file_paths])

    remaining_onnx_filenames = list(onnx_filenames.difference(trt_filenames))
    remaining_onnx_filenames.sort()
    print(f"Found {len(remaining_onnx_filenames)} models left to be converted out of total {len(onnx_file_paths)}.")

    return remaining_onnx_filenames

def convert_to_tensorrt_parallel(onnx_models_dir, trt_output_dir, num_files_to_convert):
    remaining_onnx_filenames = filter_generated_models(onnx_models_dir, trt_output_dir)
    num_files_to_convert = min(num_files_to_convert, len(remaining_onnx_filenames))
    remaining_onnx_filenames = remaining_onnx_filenames[:num_files_to_convert]
    print(f"Converting {len(remaining_onnx_filenames)} ONNX files.")
    
    export_configs = [(onnx_filename, onnx_models_dir, trt_output_dir) for onnx_filename in remaining_onnx_filenames]
    pool = mp.Pool(mp.cpu_count())
    pool.map(convert_to_tensorrt, export_configs)

def convert_to_tensorrt_sequential(onnx_models_dir, trt_output_dir, num_files_to_convert):
    remaining_onnx_filenames = filter_generated_models(onnx_models_dir, trt_output_dir)
    num_files_to_convert = min(num_files_to_convert, len(remaining_onnx_filenames))
    remaining_onnx_filenames = remaining_onnx_filenames[:num_files_to_convert]
    print(f"Converting {len(remaining_onnx_filenames)} ONNX files.")

    for onnx_filename in tqdm(remaining_onnx_filenames):
        export_configs = (onnx_filename, onnx_models_dir, trt_output_dir)
        convert_to_tensorrt(export_configs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=27, dest="seed", help="Seed value for random generator.")
    parser.add_argument("--export_dir", default="/home/saeejith/work/nas/hw-nats-bench/datasets", dest="export_dir", help="Output directory to store exported TensorRT files.")
    parser.add_argument("--input_dir", default="/home/saeejith/work/nas/hw-nats-bench/datasets/onnx-simplified", dest="input_dir", help="Input directory containing ONNX models.")
    parser.add_argument("--range", nargs=2, default=[0, 15625], dest="range", type=(int), help="Range of architectures to export")
    parser.add_argument('--convert_parallel', dest='convert_parallel', action='store_true')
    parser.add_argument('--num_models_to_convert', default=15625, dest='num_models_to_convert', type=(int), help="Number of models to convert")

    args = parser.parse_args()
    assert (len(args.range) == 2)

    gen_utils.setup_seed(args.seed)

    # Execute conversion.
    if args.convert_parallel:
        convert_to_tensorrt_parallel(args.input_dir, args.export_dir, args.num_models_to_convert)
    else:
        convert_to_tensorrt_sequential(args.input_dir, args.export_dir, args.num_models_to_convert)
