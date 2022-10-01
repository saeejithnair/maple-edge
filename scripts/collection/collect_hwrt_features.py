import argparse
from nats_bench import create
import numpy as np

from maple.utils import nasbench201_utils
from maple.collection.recorder.recorder_configs import LatencyRecorderConfig
from maple.collection.recorder.latency_recorder_manager import (
    LatencyRecorderManager)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", default=224,
                        help="Input size for model", type=int)
    parser.add_argument(
        "--range", nargs=2, default=[0, 15625], dest="range", type=(int),
        help="range of architectures measure (--range 0 15625)")
    parser.add_argument("--gpu", default=False, dest="use_gpu",
                        action='store_true')
    parser.add_argument("--enable_perf", default=False, dest="enable_perf",
                        action='store_true')
    parser.add_argument("--inference_engine", default="trt",
                        dest="inference_engine", help="")
    parser.add_argument(
        "--model_dir",
        default="/home/saeejith/work/nas/hw-nats-bench/datasets/v0.1",
        dest="model_dir", help="Directory containing serialized models.")
    parser.add_argument(
        "--output_fname", default="hwrt_features", dest="output_fname",
        help="Name of output file to be generated (without extension)."
    )
    parser.add_argument(
        "--num_runs", default=50, dest="num_runs", type=(int),
        help="Number of runs for each end to end arch latency measurement."
    )
    parser.add_argument(
        "--nats_dir", default="/home/saeejith/work/nas/NATS", dest="nats_dir",
        help="Path to dir containing NATS-tss-v1_0-3ffb9-simple dataset"
    )
    parser.add_argument(
        "--output_dir", default="/home/saeejith/work/nas/hw-nats-bench",
        dest="output_dir", help="Directory to output latency measurements.")

    return parser



def main(args):
    device = 'gpu' if args.use_gpu else 'cpu'
    assert(len(args.range) == 2)
    target_archs = np.arange(args.range[0], args.range[1])

    recorder_config = LatencyRecorderConfig(
        inference_engine=args.inference_engine,
        online=False,
        n_runs=args.num_runs,
        input_size=args.input_size,
        device=device,
        target_archs=target_archs,
        target_ops='all',
        output_dir=args.output_dir,
        model_dir=args.model_dir
    )

    # Create NATS bench API. Download from:
    # https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU
    api = create(f"{args.nats_dir}/NATS-tss-v1_0-3ffb9-simple", 'tss',
                 fast_mode=True, verbose=False)
    experiment_manager = LatencyRecorderManager(recorder_config, api)


if __name__ == "__main__":
    parser = setup_parser()
    main(parser.parse_args())
