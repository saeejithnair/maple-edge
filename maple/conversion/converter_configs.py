from collections import namedtuple

# Class for storing export configurations for converting NATS-Bench
# architectures.
ExportConfig = namedtuple(
    'ExportConfig',
    'cell_config_dict arch_idx out_dir input_shape channels_last')

# Stores the extension and the regex match string used for parsing the unique
# identifier for each model (associated with architecture index.)
FileConfig = namedtuple('FileConfig', 'extension re_uid_match dirname')

CELL_FILE_CONFIGS = {
    'onnx': FileConfig(
        extension='onnx', re_uid_match='.*nats_cell_(\d+).onnx',
        dirname='onnx/cells'),
    'onnx-simplified': FileConfig(
        extension='onnx', re_uid_match='.*nats_cell_(\d+).onnx',
        dirname='onnx-simplified/cells'),
    'keras': FileConfig(
        extension='h5', re_uid_match='.*nats_cell_(\d+).h5',
        dirname='keras/cells'),
    'tflite': FileConfig(
        extension='tflite', re_uid_match='.*nats_cell_(\d+).tflite',
        dirname='tflite/cells'),
    'torch': FileConfig(
        extension='pth', re_uid_match='.*nats_cell_(\d+).pth',
        dirname='torch/cells'),
    'trt': FileConfig(
        extension='engine', re_uid_match='.*nats_cell_(\d+).engine',
        dirname='trt/cells'),
    'torchscript': FileConfig(
        extension='pt', re_uid_match='.*nats_cell_(\d+).pt',
        dirname='torchscript/cells'),
}

OPS_FILE_CONFIGS = {
    'onnx': FileConfig(
        extension='onnx',
        re_uid_match='.*nats_ops_(.*).onnx',
        dirname='onnx/ops'
    ),
    'onnx-simplified': FileConfig(
        extension='onnx', 
        re_uid_match='.*nats_op_(.*).onnx',
        dirname='onnx-simplified/ops'
    ),
    'keras': FileConfig(
        extension='h5', 
        re_uid_match='.*nats_ops_(.*).h5',
        dirname='keras/ops'
    ),
    'tflite': FileConfig(
        extension='tflite', 
        re_uid_match='.*nats_ops_(.*).tflite',
        dirname='tflite/ops'
    ),
    'torch': FileConfig(
        extension='pth',
        re_uid_match='.*nats_ops_(.*).pth',
        dirname='torch/ops'
    ),
    'trt': FileConfig(
        extension='engine',
        re_uid_match='.*nats_ops_(.*).engine',
        dirname='trt/ops'
    ),
    'torchscript': FileConfig(
        extension='pt',
        re_uid_match='.*nats_ops_(.*).pt',
        dirname='torchscript/ops'
    ),
}

# reduction, inC and outC
NATS_CELL_CONFIG = [(1, 16, 16), (2, 32, 32), (4, 64, 64)]
# Fix input size to 224x224 to be backwards compatible with
# previously collected data.
# TODO(snair): Remember to make this dynamic because we want to support
# multiple different input sizes.
INPUT_SIZE = 224

CELL_FILE_PREFIX = 'nats_cell'
OPS_FILE_PREFIX = 'nats_ops'

TORCHSCRIPT_OPS_INPUT_SHAPE = {
    # (batch_size, channels, width, height)
    'avg_pool_3x3_112_32': (1, 32, 112, 112),
    'avg_pool_3x3_224_16': (1, 16, 224, 224),
    'avg_pool_3x3_56_64': (1, 64, 56, 56),

    'none_112_32': (1, 32, 112, 112),
    'none_224_16': (1, 16, 224, 224),
    'none_56_64': (1, 64, 56, 56),

    'nor_conv_1x1_112_32': (1, 32, 112, 112),
    'nor_conv_1x1_224_16': (1, 16, 224, 224),
    'nor_conv_1x1_56_64': (1, 64, 56, 56),

    'nor_conv_3x3_112_32': (1, 32, 112, 112),
    'nor_conv_3x3_224_16': (1, 16, 224, 224),
    'nor_conv_3x3_56_64': (1, 64, 56, 56),

    'skip_connect_112_32': (1, 32, 112, 112),
    'skip_connect_224_16': (1, 16, 224, 224),
    'skip_connect_56_64': (1, 64, 56, 56),

    'stem': (1, 3, 224, 224),
    'resblock1': (1, 16, 224, 224),
    'resblock2': (1, 32, 112, 112),
    'pool': (1, 64, 56, 56),
    'lastact': (1, 64, 56, 56),
    'classifier': (1, 64),
}
