from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Add, concatenate


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    y = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    y = BatchNormalization(name=bn_name_base + '2a')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(y)
    y = BatchNormalization(name=bn_name_base + '2b')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(y)
    y = BatchNormalization(name=bn_name_base + '2c')(y)

    y = Add()([y, input_tensor])
    y = Activation('relu')(y)
    return y


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    y = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    y = BatchNormalization(name=bn_name_base + '2a')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(y)
    y = BatchNormalization(name=bn_name_base + '2b')(y)
    y = Activation('relu')(y)

    y = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(y)
    y = BatchNormalization(name=bn_name_base + '2c')(y)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    y = Add()([y, shortcut])
    y = Activation('relu')(y)
    return y


def ResNet50(input_shape, refinenet=False, deeplab=False):
    """Instantiates the ResNet50 architecture.
    # Arguments
    input_shape: tuple 
        Input image shape with chanels
    refinenet: bool, optional
        Is this ResNet50 used in RefineNet
    deeplab: bool, optional
        Is this ResNet50 used in DeepLabV3
    # Returns
        A tensorflow.keras.Model instance of ResNet50
    """
    img_input = Input(input_shape)

    out = []

    x = ZeroPadding2D((2, 2))(img_input)
    x = Conv2D(64, (5, 5), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    if refinenet:
        out.append(x)

    if deeplab:
        out.append(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    if refinenet:
        out.append(x)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    if refinenet:
        out.append(x)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    out.append(x)

    return Model(img_input, out)


def conv_block_DSPP(input, n_filters=256, kernel_size=3, dilation_rate=1):
    """
    Convolutional block - part of Dilated Spatial Pyramid Pooling module.
    # Arguments:
    input: Keras tensor
        Tensor from previous layer
    n_filters: int, optional
        Channel number of output feature map
    kernel_size: int, optional
        Size of convolutional kernel
    dilation_rate: int, optional
        Dilation rate used only in DSPP block
    # Returns:
        Keras tensor output of conv block
    """
    x = Conv2D(n_filters, kernel_size, dilation_rate=dilation_rate,
        padding="same", use_bias=False)(input)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def DSPP(input):
    """"
    Dilated Spatial Pyramid Pooling module (DSPP) - part of DeepLabv3
    # Arguments:
    input: Keras tensor
        Tensor from previous layer
    # Returns:
        Keras tensor output of DSPP block
    """
    input_shape = input.shape
    x = AveragePooling2D(pool_size=(input_shape[-3], input_shape[-2]))(input)
    x = conv_block_DSPP(x, kernel_size=1)
    out_pool = UpSampling2D(size=(input_shape[-3] // x.shape[1], input_shape[-2] // x.shape[2]), 
        interpolation="bilinear")(x)
    out_1 = conv_block_DSPP(input, kernel_size=1, dilation_rate=1)
    out_6 = conv_block_DSPP(input, kernel_size=3, dilation_rate=6)
    out_12 = conv_block_DSPP(input, kernel_size=3, dilation_rate=12)
    out_18 = conv_block_DSPP(input, kernel_size=3, dilation_rate=18)

    x = concatenate([out_pool, out_1, out_6, out_12, out_18])
    return conv_block_DSPP(x, kernel_size=1)


def DeepLabV3(input_shape, nb_classes):
    """
    Build DeepLabV3+ semantic segmentation model with ResNet50 as backbone
    # Arguments:
    input_shape: tuple
        Tuple of input image shape (width, height, number_of_channels)
    nb_classes: int
        Number of classes used in segmentation data
    # Returns:
        Tensorflow.keras.Model of DeepLabV3
    """
    res = ResNet50(input_shape, deeplab=True)  # backbone model
    input = res.input  # get input
    res_outs = res.output  # get ResNet50 outputs
    x = res_outs[1]  # main output W // 32, H // 32
    x = DSPP(x)  # build DSPP block

    x = UpSampling2D(size=(input_shape[0] / 4 // x.shape[1], input_shape[1] / 4 // x.shape[2]),
        interpolation="bilinear")(x)  # upsample by 4
    y = res_outs[0]  # get output of W // 4, H // 4
    y = conv_block_DSPP(y, kernel_size=1)
    x = concatenate([x, y])  # concat 2 feature maps
    x = conv_block_DSPP(x)
    x = conv_block_DSPP(x)
    x = UpSampling2D(size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]),
        interpolation="bilinear")(x)  # upsample to original shape

    x = Conv2D(nb_classes, 1, padding="same")(x)  # build output feature map and apply activation
    x = Activation('softmax')(x)
    return Model(input, x)
