# Import all modules for model building
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, ZeroPadding2D, Lambda, Conv3D, MaxPool3D
from tensorflow.keras.layers import Concatenate, Add, Layer, Multiply, Reshape
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K




def BN(name=""):
    """
    Batch normalization layer
    """
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    """
    Residual short used in the beggining of each lvl
    """
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


class Interp(Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = tf.image.resize(inputs, [new_height, new_width])
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config
    

def ResNet_with_different_level_features(inp, layers):
    
    # inp - Input layer
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    layers_to_merge = []  # this layers will be concatenated

    # Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', 
                  name=names[0], use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                  name=names[2], use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', 
                  name=names[4], use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    layers_to_merge.append(res)  # first part of the network will be concatenated with all next

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    layers_to_merge.append(res)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)

    layers_to_merge.append(res)

    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    layers_to_merge.append(res)

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)

    # applying activations to all the layers
    layers_to_merge[0] = Activation('relu')(layers_to_merge[0])
    layers_to_merge[1] = Activation('relu')(layers_to_merge[1])
    layers_to_merge[2] = Activation('relu')(layers_to_merge[2])
    layers_to_merge[3] = Activation('relu')(layers_to_merge[3])

    # resizing first two layers to match all the remaining
    layers_to_merge[0] = Interp([60, 60])(layers_to_merge[0])
    layers_to_merge[1] = Interp([60, 60])(layers_to_merge[1])

    layers_to_merge[2] = Conv2D(128, kernel_size=(1, 1))(layers_to_merge[2])

    layers_to_merge.append(res)
    final = Concatenate()(layers_to_merge)

    final = Conv2D(filters=2048, kernel_size=(1, 1))(final)

    return final
  
  
def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (473, 473):
        kernel_strides_map = {1: 60,
                              2: 30,
                              3: 20,
                              6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90,
                              2: 45,
                              3: 30,
                              6: 15}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer

def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res
  
  
def CPA_net(nb_classes, resnet_layers, input_shape, activation='softmax'):
    
    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet_with_different_level_features(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    xx = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    xx = BN(name="conv5_4_bn")(xx)
    xx = Activation('relu')(xx)
    xx = Dropout(0.1)(xx)

    # channel weighting starts
    channel_weighting = MaxPooling2D((60, 60))(xx)
    channel_weighting = Dense(256)(channel_weighting)
    channel_weighting = Activation('relu')(channel_weighting)
    channel_weighting = Dropout(0.1)(channel_weighting)
    channel_weighting = Dense(512)(channel_weighting)
    channel_weighting = Activation('sigmoid')(channel_weighting)
    # channel_weighting = Dropout(0.1)(channel_weighting)
    xx = Multiply()([xx, channel_weighting])
    # channel weighting ends

    # pixel attention starts
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(xx)
    pixel_weighting = BN(name="pixel_attention_conv_bn_1")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_2")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Conv2D(512, (1, 1), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_3")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(2, (1, 1))(pixel_weighting)

    pixel_weighting_multiply = Activation('relu')(pixel_weighting)
    pixel_weighting_multiply = Conv2D(1, (1, 1))(pixel_weighting_multiply)
    pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting_multiply)
    xx = Multiply()([xx, pixel_weighting_multiply])
    xx = Add()([xx, pixel_weighting_multiply])

    pixel_weighting_output = Interp([input_shape[0], input_shape[1]])(pixel_weighting)
    # pixel attention ends

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(xx)
    
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = build_CPA_net(inp, x, pixel_weighting_output)

    return model


def build_CPA_net(input, output, mask_output):
    """
    input - Input
    output - main feature map output
    mask_output - mask for pixel-wise attention
    """
    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape  # output shape
    i_shape = Model(img_input, o).input_shape  # input shape

    output_height = o_shape[1]  # output dimensions
    output_width = o_shape[2]
    input_height = i_shape[1]  # input dimensions
    input_width = i_shape[2]
    n_classes = o_shape[3]  # number of classes

    o = (Reshape((output_height * output_width, -1)))(o)  # reshape main output (output_height * output_width, n_classes)
    mask_output = (Reshape((output_height * output_width, -1)))(mask_output)  # same for mask output

    o = (Activation('softmax', name="main_output_activation"))(o)  # apply activation on main output
    mask_output = (Activation('softmax', name="mask_output_activation"))(mask_output)  # apply activation on output mask

    model = Model(img_input, [o, mask_output])  # build model from input and 2 outputs
    model.output_width = output_width  # set model's params
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = "CPAnet"
    
    return model  
