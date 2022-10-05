# For your non-Colab code, be sure you have tensorflow==1.15
# TODO(snair): Format this file properly.
import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Conv2D,
    AveragePooling2D,
    ZeroPadding2D,
    Add,
    Multiply,
    BatchNormalization,
    Lambda
)
from tensorflow.keras.regularizers import l2

from copy import deepcopy

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


def BNReLu(inputs):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(inputs)
    return Activation("relu")(norm)


def ReLUConvBN(C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    """Helper to build a relu block - relu -> conv -> BN  block
    """
    assert track_running_stats==True, "track_running_stats has to be True"

    def f(inputs):
        act = Activation("relu")(inputs)
        conv = Conv2D(filters=C_out, kernel_size=(kernel_size,kernel_size),
                      strides=(stride,stride), padding=padding,
                      dilation_rate=(dilation,dilation),
                      use_bias=not affine,
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(1.e-4))(act)
        norm = BatchNormalization(axis=CHANNEL_AXIS, center=affine, scale=affine)(conv)
        return norm

    return f

def ResNetBasicblock(planes, stride, affine=True, track_running_stats=True):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(inputs):
        assert stride == 2, 'invalid stride {:}'.format(stride)
        inputs_padded = ZeroPadding2D(1)(inputs)
        conv_a = ReLUConvBN(planes, 3, stride, "valid", 1, affine, track_running_stats)(inputs_padded)
        conv_b = ReLUConvBN(planes, 3,      1, "same", 1, affine, track_running_stats)(conv_a)

        downsample = AveragePooling2D(pool_size=(2,2))(inputs)
        residual = Conv2D(filters=planes, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False)(downsample)
        
        return Add()([conv_b, residual])

    return f

def Zero(C_out, stride):
    assert stride == 1, "Zero must use stride 1"
    def f(inputs): # I tried Mul, Sub, Zeros, all failed at the Edge TPU complier part, so I replace it with Identity
        return inputs

    return f

def Identity():
    def f(inputs):
        return inputs

    return f

def BackboneStem(C):
    def f(inputs):
        block = Conv2D(filters=C, kernel_size=(3, 3), padding="same", use_bias=False)(inputs)
        block = BatchNormalization(axis=CHANNEL_AXIS)(block)

        return block

    return f

def BackboneBNReLu():
    def f(inputs):
        return BNReLu(inputs)

    return f

def BackboneClassifier(N):
    def f(inputs):
        block = Flatten()(inputs)
        return Dense(units=N, kernel_initializer="he_normal")(block)

    return f

OPS = {
  'none'         : lambda C_out, stride, affine, track_running_stats: Zero(C_out, stride),
  'avg_pool_3x3' : lambda C_out, stride, affine, track_running_stats: AveragePooling2D((3,3), stride, padding="same"), 
  'nor_conv_3x3' : lambda C_out, stride, affine, track_running_stats: ReLUConvBN(C_out, 3, stride, "same", 1, affine, track_running_stats),
  'nor_conv_1x1' : lambda C_out, stride, affine, track_running_stats: ReLUConvBN(C_out, 1, stride, "same", 1, affine, track_running_stats),
  'skip_connect' : lambda C_out, stride, affine, track_running_stats: Identity(),
}

def InferCell(genotype, C_out, stride, affine=True, track_running_stats=True):
    assert stride == 1, "InferCell must use stride 1"
    layers = []
    node_IN = []
    node_IX = []
    genotype = deepcopy(genotype)

    for i in range(1, len(genotype)):
        node_info = genotype[i-1]
        cur_index = []
        cur_innod = []
        for (op_name, op_in) in node_info:
            if op_in == 0:
                layer = OPS[op_name](C_out, stride, affine, track_running_stats)
            else:
                layer = OPS[op_name](C_out,      1, affine, track_running_stats)
            cur_index.append( len(layers) )
            cur_innod.append( op_in )
            layers.append( layer )
        node_IX.append( cur_index )
        node_IN.append( cur_innod )
    nodes = len(genotype)
    def f(inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(zip(node_IX, node_IN)):
            output_to_sum = [layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods)]
            if len(output_to_sum) == 1:
                node_feature = output_to_sum[0]
            else:
                node_feature = Add()( output_to_sum )
            nodes.append( node_feature )
        return nodes[-1]

    return f

class Backbone():
    def __init__(self, C=16, N=10, h=224, w=224) -> None:
        self.C = C
        self.N = N
        self.h = h
        self.w = w
        self.torch_device = 'cpu'

        self._setup_static_inputs(self.h, self.w)
        self._setup_static_blocks(self.C, self.N, self.h, self.w)

    def _setup_static_blocks(self, C, N, h, w):
        self.blocks = {
            'stem': BackboneStem(C),
            'resblock1': ResNetBasicblock(C*2, 2, True),
            'resblock2': ResNetBasicblock(C*4, 2, True),
            'pool': AveragePooling2D(pool_size=(h//4, w//4), strides=(1, 1)),
            'lastact': BackboneBNReLu(),
            'classifier': BackboneClassifier(N),
        }
    
    def _setup_static_inputs(self, h, w):
        self.inputs = {'stem': Input(shape=(w,h,3), dtype=tf.float32, batch_size=1),
                    'resblock1': Input(shape=(w,h,self.C), dtype=tf.float32, batch_size=1),
                    'resblock2': Input(shape=(w//2,h//2,self.C*2), dtype=tf.float32, batch_size=1),
                    'pool': Input(shape=(w//4,h//4,self.C*4), dtype=tf.float32, batch_size=1),
                    'lastact': Input(shape=(w//4,h//4,self.C*4), dtype=tf.float32, batch_size=1),
                    'classifier': Input(shape=(self.C*4), dtype=tf.float32, batch_size=1),
                    }

    def get_block_names(self):
        return list(self.blocks.keys())
    
    def get_block(self, block_name):
        return self.blocks[block_name]

    def get_input(self, block_name):
        return self.inputs[block_name]


def TinyNetwork(C, N, genotype, num_classes, input_size):
    # Hardcode input shape to 224x224
    inputs_shape = (input_size, input_size,3)
    inputs = Input(shape=inputs_shape)
    stem = BatchNormalization(axis=CHANNEL_AXIS)((Conv2D(filters=C, kernel_size=(3, 3), padding="same", use_bias=False)(inputs)))
    block = stem 
    
    # Set Repeated blocks here
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
        if reduction:
            block = ResNetBasicblock(C_curr,2,True)(block)
        else:
            block = InferCell(genotype,C_curr,1)(block)
        
    block = BNReLu(block)
    block_shape = K.int_shape(block)
    global_pooling = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(block)
    flatten = Flatten()(global_pooling)
    dense = Dense(units=num_classes, kernel_initializer="he_normal")(flatten)
    model = Model(inputs=inputs, outputs=dense)
    return model

def OpsNetwork(op, inputs_shape):
    batch_size, w, h, c = inputs_shape
    print(inputs_shape)
    inputs = Input(shape=(w,h,c), batch_size=batch_size)
    print(inputs.shape)
    outputs = op(inputs)
    # return op
    return Model(inputs=inputs, outputs=outputs)

def OpsBackboneNetwork(op, inputs):
    print(inputs.shape)
    outputs = op(inputs)
    return Model(inputs=inputs, outputs=outputs)
