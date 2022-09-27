from typing import Optional
import numpy as np
import time
import torch
from maple.utils import nasbench201_utils as nasbench201_utils
from dataclasses import dataclass
import csv


@dataclass
class ArchLatencyMeasure:
    nasbench201_idx: int
    end2end: float
    summed: float
    encoding: np.array
    arch_str: list
    # Look up table
    lut: np.array
    op_latency: np.array
    end2end_arr: Optional[np.array] = None


def calc_latency(LUT, arch_str) -> float:
    """Calculates total latency by summing up measured latency of blocks/layers.

    Args:
        LUT: Dictionary of {layer/block names: measured latency}.
        arch_str: Encodes network architecture topology.

    Returns:
        Total latency.

    TODO: Make op2key call not specific to 224 size input.
    """
    ops = nasbench201_utils.decode_arch_str(arch_str)
    cell_config = [
        (1, 16, 16),
        (2, 32, 32),
        (4, 64, 64)]
    blocks = [
                'input', 'resblock1', 'resblock2',
                'pool', 'lastact', 'classifier']
    total_l = 0

    # Sum up latency of blocks in LUT.
    for block in blocks:
        total_l += LUT[block]

    for (reduction, _, outC) in cell_config:
        cell_latency = 0.0
        for op in ops:
            key = nasbench201_utils.op2key(op, 224//reduction, outC)
            cell_latency += LUT[key]
        total_l += cell_latency*5

    return total_l


def _latency_on_cpu(net, inputs, N, context) -> list[float]:
    """Measures inference latency of a network on CPU for N runs.

    Args:
        net: PyTorch model.
        inputs: Dummy inputs to be passed into model.
        N: Number of inference trials to run.
        context: Specifies which context manager should be used
                 {'none', 'no_grad', 'inference_mode'}.

    Returns:
        List of size N containing latencies.
    """
    net.to('cpu')
    net.eval()
    lat = []

    for _ in range(5):
        _ = net(inputs)
    if context == 'none':
        from contextlib import nullcontext
        cm = nullcontext()
    elif context == 'no_grad':
        cm = torch.no_grad()
    elif context == 'inference_mode':
        cm = torch.inference_mode()
    else:
        raise ValueError("Unknown context mode")

    with cm:
        for n in range(N):
            start = time.time()
            _ = net(inputs)
            end = time.time()
            lat.append(end-start)

    lat = np.array(lat)
    return lat


def _latency_on_gpu(net, inputs, N=100, context='none'):
    """Measures inference latency of a network on GPU for N runs.

    Args:
        net: PyTorch model.
        inputs: Dummy inputs to be passed into model.
        N: Number of inference trials to run.
        context: Specifies which context manager should be used
                 {'none', 'no_grad', 'inference_mode'}.

    Returns:
        Array of size N containing latencies.
    """
    net.to('cuda:0')
    net.eval()
    timings = np.zeros((N, 1))
    starter, ender = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
    for _ in range(50):
        _ = net(inputs)

    if context == 'none':
        from contextlib import nullcontext
        cm = nullcontext()
    elif context == 'no_grad':
        cm = torch.no_grad()
    elif context == 'inference_mode':
        cm = torch.inference_mode()
    else:
        raise ValueError("Unknown context mode")

    with cm:
        for n in range(N):
            starter.record()
            _ = net(inputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[n] = curr_time/1000

    return timings


def latency(net, inputs, device, N=100, context='none'):
    if device == 'cpu':
        return _latency_on_cpu(net, inputs, N, context)

    if device == 'cuda' or device == 'cuda:0':
        return _latency_on_gpu(net, inputs, N, context)


def read_perf_csv(fname):
    with open(fname) as csvfile:
        perfreader = csv.reader(csvfile)
        records = []
        for row in perfreader:
            if row:
                if row[0][0] == '#':
                    rec = {}
                elif len(row) == 3 or len(row) == 2:
                    rec['op'] = row[0]
                    rec['l'] = float(row[1])
                    records.append(rec)
                else:
                    try:
                        key = row[2]
                    except:
                        print(row)
                    try:
                        rec[key] = int(row[0])
                    except ValueError:
                        # Set to NaN if operation wasn't supported on device.
                        rec[key] = np.NaN
        return records
