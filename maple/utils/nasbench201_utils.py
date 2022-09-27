import numpy as np

# One-hot encoding of supported operations.
OPS = {'avg_pool_3x3': np.array([0, 1, 0, 0]),
       'nor_conv_1x1': np.array([0, 0, 0, 1]),
       'skip_connect': np.array([1, 0, 0, 0]),
       'nor_conv_3x3': np.array([0, 0, 1, 0]),
       'none': np.array([0, 0, 0, 0])}


def op2key(op: str, input_size: int, out_channels: int) -> str:
    """Convert op to hashable string that can be used as a key.

    Args:
        op: Operation code (e.g. {'avg_pool_3x3', 'nor_conv_1x1', ...}).
        input_size: Size of input dimension (assume input is cropped square).
        out_channels: Number of output channels
    """
    return f"{op}_{input_size}_{out_channels}"


def decode_arch_str(arch_str: str) -> list[str]:
    """Converts architecture encoding to list of operations.

    Args:
        arch_str: Encodes network architecture topology.
    """
    cell_ops = []
    node_strs = arch_str.split('+')

    for node_str in node_strs:
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs:
            assert len(xinput.split('~')
                       ) == 2, 'invalid input length : {:}'.format(xinput)
        for xi in inputs:
            op, idx = xi.split('~')
            if op not in OPS:
                raise ValueError()
            cell_ops.append(op)
    return cell_ops


def str2encoding(arch_str: str) -> np.ndarray:
    """Converts architecture encoding from string representation to list of
    one-hot encodings.

    Args:
        arch_str: Encodes network architecture topology.
    """
    node_strs = arch_str.split('+')
    encoding = []
    for node_str in node_strs:
        inputs = list(filter(lambda x: x != '', node_str.split('|')))
        for xinput in inputs:
            assert len(xinput.split('~')
                       ) == 2, 'invalid input length : {:}'.format(xinput)
        for xi in inputs:
            op, idx = xi.split('~')
            if op not in OPS:
                raise ValueError()
            op_encoding = OPS[op]
            encoding.append(op_encoding)

    return np.array(encoding).flatten()
