import argparse
import keras
import os
import numpy as np
from nats_bench import create
import sys
import tensorflow as tf
from xautodl import config_utils
from xautodl.models.cell_operations import NAS_BENCH_201

from maple.conversion import converter
from maple.conversion import converter_configs as cc
from maple.conversion.converter_configs import (CELL_FILE_CONFIGS,
                                                OPS_FILE_CONFIGS)
from maple.utils import utils

from maple.cells.tiny_network_keras import TinyNetwork as TinyNetworkKeras
from maple.cells.tiny_network_keras import OpsBackboneNetwork
from maple.cells.tiny_network_keras import Backbone as BackboneKeras
from maple.cells.tiny_network_keras import OpsNetwork as OpsNetworkKeras
from maple.cells.tiny_network_keras import OPS as OPS_KERAS
from keras.utils.vis_utils import plot_model

"""
Tensorflow Lite model conversion script.

This script asynchronoulsy converts all the NATS-Bench-201 cell architectures
to Tensorflow Lite models. The TFLite models are saved to the specified
output directory. The script also validates that the exported TFLite models
are correct by comparing the output of the exported TFLite models to the
output of the original Keras networks.

Pipeline execution sequence works as follows:
1. Load NATS-Bench-201 dataset/create NATS-Bench-201 API.
2. For each cell architecture in the dataset:
    a. Convert NATS cell to Keras network.
    b. Save Keras network to disk.
    c. Validate that the exported Keras model is correct.
    d. Convert Keras network to Tensorflow Lite model.
    e. Save Tensorflow Lite model to disk.
    f. Validate that the exported Tensorflow Lite model is correct.
3. Convert NATS cell ops to TFLite networks.
4. Convert NATS cell backbone to TFLite network.
"""


def get_keras_network(config, input_size):
    """Returns the Keras network given a NATS cell architecture index."""
    genotype = converter.get_nats_cell(config)

    # Convert NATS cell to Keras network
    keras_net = TinyNetworkKeras(config.C, config.N, genotype,
                                 config.num_classes, input_size)
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.,
                                                      seed=27)

    for layer in keras_net.layers:
        layer.set_weights(
            [initializer(shape=w.shape) for w in layer.get_weights()])

    keras_net.compile(optimizer="SGD", loss="categorical_crossentropy")

    return keras_net


def get_ops_network(op_name, C_in, C_out, inputs_shape):
    """Returns the Keras network given a NATS cell op name.

    Args:
        op_name: Name of the op to convert.
        C_in: Number of input channels (unused).
        C_out: Number of output channels.
        inputs_shape: Shape of the input tensor.
    """
    op_net = OPS_KERAS[op_name](C_out, 1, True, True)
    print(f'Creating op network {op_name}')
    return OpsNetworkKeras(op_net, inputs_shape)


def validate_keras_model(keras_net, exported_keras_net_path, dummy_input,
                         generate_model_png=False):
    """Validates that the exported Keras model is correct.

    Args:
        keras_net: Keras network to validate.
        exported_keras_net_path: Path to the exported Keras model.
        dummy_input: Dummy input to the network.

    Raises:
        ValueError: If the output of the exported Keras model does not match
                    the output of the original Keras network.
    """
    exported_keras_net = keras.models.load_model(exported_keras_net_path)
    exported_keras_net_output = exported_keras_net.predict(dummy_input)
    keras_net_output = keras_net.predict(dummy_input)

    if not np.allclose(keras_net_output, exported_keras_net_output,
                       rtol=1e-03, atol=1e-05):
        if generate_model_png:
            plot_model(keras_net, to_file=f'{exported_keras_net_path}.png',
                       show_shapes=True, show_layer_names=True)
        raise ValueError(
            f"Validation failed. Output for exported Keras model "
            f"{exported_keras_net_path} does not match output of "
            f"original Keras network.")


def validate_tflite_model(keras_net, exported_tflite_net_path, dummy_input,
                          generate_model_png=False):
    """Validates that the exported Tensorflow Lite model is correct.

    Args:
        keras_net: Keras network to validate.
        exported_tflite_net_path: Path to the exported Tensorflow Lite model.
        dummy_input: Dummy input to the network.

    Raises:
        ValueError: If the output of the exported Tensorflow Lite model does
                    not match the output of the original Keras network.
    """
    interpreter = tf.lite.Interpreter(model_path=exported_tflite_net_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    tflite_net_output = interpreter.get_tensor(output_details[0]['index'])

    keras_net_output = keras_net.predict(dummy_input)

    if not np.allclose(keras_net_output, tflite_net_output, rtol=1e-03,
                       atol=1e-05):
        if generate_model_png:
            plot_model(
                keras_net, to_file=f'{exported_tflite_net_path}.png',
                show_shapes=True, show_layer_names=True)

        raise ValueError(
            f"Validation failed. Output for exported TfLite model "
            f"{exported_tflite_net_path} does not match output of "
            f"original Keras network.")


def convert_keras_to_tflite(keras_net, keras_model_out_path,
                            tflite_model_out_path, dummy_input):
    """Converts a Keras model to Tensorflow Lite.

    Args:
        keras_net: Keras network to convert.
        keras_model_out_path: Path to save the exported Keras model.
        tflite_model_out_path: Path to save the exported Tensorflow Lite model.
        dummy_input: Dummy input to the network.
    """
    # Export Keras network to disk in .h5 format
    keras_net.save(keras_model_out_path)

    # Validate that export happened correctly.
    validate_keras_model(keras_net, keras_model_out_path, dummy_input)
    print(f'Exported and validated Keras model {keras_model_out_path}')

    # Convert Keras model to Tensorflow Lite
    tflite_engine = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(
                        keras_model_out_path)
    tflite_model = tflite_engine.convert()

    with open(tflite_model_out_path, "wb") as f:
        print(f"Starting to write {tflite_model_out_path} to disk.")
        sys.stdout.flush()
        f.write(tflite_model)
        print(f"Finished writing {tflite_model_out_path} to disk.")
        sys.stdout.flush()

    validate_tflite_model(keras_net, tflite_model_out_path, dummy_input)

    print(f'Exported and validated TfLite model {tflite_model_out_path}')
    sys.stdout.flush()


def convert_nats_to_tflite(export_config):
    """Converts a NATS cell to Tensorflow Lite."""
    cell_config_dict = export_config.cell_config_dict
    # Converts config from a dict to a namedtuple.
    cell_config = config_utils.dict2config(cell_config_dict, None)
    w, h = export_config.input_shape
    dummy_input = np.float32(np.random.random((1, w, h, 3)))
    dummy_target = np.float32(np.random.random((1, 10)))

    keras_net = get_keras_network(cell_config, input_size)
    keras_net.fit(dummy_input, dummy_target)

    model_uid = utils.generate_model_uid(
        input_size=export_config.input_shape[0],
        arch_idx=export_config.arch_idx)

    # Generate path for saving Keras model.
    keras_model_out_path = utils.generate_model_out_path(
                            export_dir=export_config.export_dir,
                            dirname=CELL_FILE_CONFIGS['keras'].dirname,
                            model_uid=model_uid,
                            extension=CELL_FILE_CONFIGS['keras'].extension)

    # Generate path for saving Tensorflow Lite model.
    tflite_model_out_path = utils.generate_model_out_path(
                            export_dir=export_config.export_dir,
                            dirname=CELL_FILE_CONFIGS['tflite'].dirname,
                            model_uid=model_uid,
                            extension=CELL_FILE_CONFIGS['tflite'].extension)

    # Convert to TFLite and save.
    try:
        convert_keras_to_tflite(keras_net, keras_model_out_path,
                                tflite_model_out_path, dummy_input)
    except Exception as e:
        print(f"Encountered exception while trying to convert "
              f"architecture {model_uid}")
        # An error may have been raised in another worker process while this
        # process was in the middle of exporting a model. We don't want
        # corrupt files so delete any model that may not have been successfully
        # exported and validated.
        print(e)
        sys.stdout.flush()

        if os.path.exists(tflite_model_out_path):
            os.remove(tflite_model_out_path)
        if os.path.exists(keras_model_out_path):
            os.remove(keras_model_out_path)


def convert_nats_backbone_to_tflite(export_dir, input_shape):
    """Converts the NATS-Bench-201 backbone to Tensorflow Lite
    and saves it to disk.

    Args:
        export_dir: Directory to save the exported model.
        input_shape: Shape of the input to the network.
    """
    inC = 16
    n_classes = 10
    w, h = input_shape
    print("Converting NATS backbone to TfLite.")

    backbone = BackboneKeras(inC, n_classes, w=w, h=h)

    backbone_blocks = backbone.get_block_names()
    for block_name in backbone_blocks:
        backbone_block_net = backbone.get_block(block_name)
        backbone_block_input = backbone.get_input(block_name)
        backbone_block_net = OpsBackboneNetwork(backbone_block_net,
                                                backbone_block_input)

        dummy_input = np.float32(np.random.random(backbone_block_input.shape))

        # Generate path for saving Keras model.
        model_out_path = utils.generate_model_ops_out_path(
                            export_dir=export_dir,
                            dirname=OPS_FILE_CONFIGS['keras'].dirname,
                            model_uid=block_name,
                            extension=OPS_FILE_CONFIGS['keras'].extension)

        # Generate path for saving Tensorflow Lite model.
        tflite_model_out_path = utils.generate_model_ops_out_path(
                            export_dir=export_dir,
                            dirname=OPS_FILE_CONFIGS['tflite'].dirname,
                            model_uid=block_name,
                            extension=OPS_FILE_CONFIGS['tflite'].extension)

        # Convert to TFLite and save.
        convert_keras_to_tflite(backbone_block_net, model_out_path,
                                tflite_model_out_path, dummy_input)


def convert_nats_ops_to_tflite(export_dir, net_input_shape):
    """Converts the NATS-Bench-201 ops to Tensorflow Lite.

    Args:
        export_dir: Directory to save the exported model.
        net_input_shape: Shape of the input to the network.
    """
    print('Converting NATS ops to TfLite')
    # reduction, inC (input channels) and outC (output channels)
    cell_config = [(1, 16, 16), (2, 32, 32), (4, 64, 64)]
    w, h = net_input_shape
    for op_name in NAS_BENCH_201:
        for (reduction, inC, outC) in cell_config:
            op_inputs_shape = (1, w//reduction, h//reduction, inC)
            # dummy_input = Input(shape=op_inputs_shape)
            dummy_input = np.float32(np.random.random(op_inputs_shape))

            # Assume op depth = 1
            opnet = get_ops_network(op_name, inC, outC,
                                    inputs_shape=op_inputs_shape)
            op_key = utils.get_op_key(op_name, w, reduction, outC)

            # Generate path for saving Keras model.
            model_out_path = utils.generate_model_ops_out_path(
                            export_dir=export_dir,
                            dirname=OPS_FILE_CONFIGS['keras'].dirname,
                            model_uid=op_key,
                            extension=OPS_FILE_CONFIGS['keras'].extension)

            # Generate path for saving Tensorflow Lite model.
            tflite_model_out_path = utils.generate_model_ops_out_path(
                            export_dir=export_dir,
                            dirname=OPS_FILE_CONFIGS['tflite'].dirname,
                            model_uid=op_key,
                            extension=OPS_FILE_CONFIGS['tflite'].extension)

            # Convert to TFLite and save.
            convert_keras_to_tflite(opnet, model_out_path,
                                    tflite_model_out_path, dummy_input)


def convert_nats_to_tflite_parallel(api, export_dir, arch_idx_range,
                                    input_shape, channels_last=False):
    """Converts NATS network from Keras to Tensorflow Lite. in parallel
    using multiprocessing.

    Args:
        api: NATS-Bench-201 API.
        export_dir: Directory to save the exported model.
        arch_idx_range: Range of architectures to export.
        input_shape: Shape of the input to the network.
        channels_last: Whether to use channels last format.
    """
    converter.export_nats_parallel(
                api, convert_nats_to_tflite, export_dir, arch_idx_range,
                input_shape, model_type='tflite', channels_last=channels_last)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=27, dest="seed",
        help="Seed value for random generator.")
    parser.add_argument(
        "--input_size", default=cc.DEFAULT_INPUT_SIZE, dest="input_size",
        help="Input size for models.")
    parser.add_argument(
        "--export_dir", default="/home/saeejith/work/nas/maple-data/models",
        dest="export_dir",
        help="Output directory to store exported TfLite files.")
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
    assert (len(args.range) == 2)

    # Input shape for the network must be square.
    input_size = int(args.input_size)
    input_shape = (input_size, input_size)

    utils.setup_seed(args.seed, envs=['tf'])

    # Create output directory if it doesn't exist.
    utils.create_outdirs(args.export_dir, 'keras')
    utils.create_outdirs(args.export_dir, 'tflite')

    # Create NATS-Bench API
    api = create(args.nats_dir, 'tss', fast_mode=True, verbose=False)

    # Execute conversion.
    convert_nats_to_tflite_parallel(api, args.export_dir, args.range,
                                    input_shape)

    if args.convert_ops:
        convert_nats_ops_to_tflite(args.export_dir, input_shape)

    if args.convert_backbone:
        convert_nats_backbone_to_tflite(args.export_dir, input_shape)
