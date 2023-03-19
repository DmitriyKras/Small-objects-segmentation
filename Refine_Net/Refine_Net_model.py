from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Add


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


def RCU(inputs, n_filters=256, kernel_size=3, name=''):
    """
    A local residual convolutional unit
    # Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel
    # Returns:
      Output of local residual convolutional unit
    """
    x = Activation("relu", name=name+'relu1')(inputs)
    x = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv1')(x)
    x = Activation("relu", name=name+'relu2')(x)
    x = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv2')(x)

    return Add(name=name+'sum')([x, inputs])


def CRP(inputs, n_filters=256, name=''):
    """
    Chained residual pooling aims to capture background 
    context from a large image region. This component is 
    built as a chain of 3 pooling blocks, each consisting 
    of one max-pooling layer and one convolution layer. 
    The output feature maps of all pooling blocks are 
    summed.
    # Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
    # Returns:
      Double-pooled feature maps
    """
    
    x = Activation("relu", name=name+'relu1')(inputs)
    sum1 = x
    
    x = Conv2D(n_filters, 3, padding='same', name=name+'conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool1')(x)
    sum2 = x
    
    x = Conv2D(n_filters, 3, padding='same', name=name+'conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool2')(x)
    sum3 = x
    
    x = Conv2D(n_filters, 3, padding='same', name=name+'conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool3')(x)
    sum4 = x

    return Add(name=name+'sum')([sum1, sum2, sum3, sum4])


def MRF(high_input=None, low_inputs=None, n_filters=256,name=''):
    """
    Multi-Resolution Fusion fuses together 2 path inputs. This block first applies 
    convolutions for inputs generating feature maps of the same feature dimension 
    (the smallest one among the inputs), and then up-samples all (smaller) 
    feature maps to the largest resolution of the inputs. Finally, all features 
    maps are summed.
    # Arguments:
      high_input: The input tensor that have the highest resolution
      low_inputs: List of the input tensors that have lower resolution in order
      of decreasing resolution
      n_filters: Number of output feature maps for each conv
    # Returns:
      Fused feature maps at higher resolution
    """
    if low_inputs is None:
        return high_input

    conv_high = Conv2D(n_filters, 3, padding='same', name=name +'conv_hi')(high_input)
    conv_high = BatchNormalization()(conv_high)
    upsampled = []

    for i, low_input in enumerate(low_inputs):
        conv_low = Conv2D(n_filters, 3, padding='same', name=name +'conv_lo_{}'.format(i))(low_input)
        conv_low = BatchNormalization()(conv_low)
        conv_low_up = UpSampling2D(size=2 ** (i + 1), interpolation='bilinear', name=name +'up_{}'.format(i))(conv_low)
        upsampled.append(conv_low_up)

    upsampled.append(conv_high)

    return Add(name=name+'sum')(upsampled)


def RefineBlock(high_input=None, low_inputs=None, block=0):
    """
    A RefineNet Block which combines together the ResidualConvUnits (RCU),
    Multi-Resolution Fusion (MRF) and then gets large-scale context with 
    the ResidualConvUnit.
    # Arguments:
      high_input: The input tensor that have the higher resolution
      low_inputs: List of the input tensors that have lower resolution in order
      of decreasing resolution
    # Returns:
      RefineNet block for a single path i.e one resolution
    """
    high_n = high_input.shape[-1]
    rcu_high = RCU(high_input, n_filters = high_n, name='rb_{}_rcu_h1_'.format(block))
    rcu_high = RCU(rcu_high, n_filters = high_n, name='rb_{}_rcu_h2_'.format(block))

    if low_inputs is None:
        chain_pooling = CRP(rcu_high, n_filters = high_n, name='rb_{}_crp_'.format(block))
        output = RCU(chain_pooling, n_filters = high_n, name='rb_{}_rcu_o1_'.format(block))
        return output

    for i in range(len(low_inputs)):
        low_n = low_inputs[i].shape[-1]
        rcu_low = RCU(low_inputs[i], n_filters = low_n, name='rb_{}_rcu_l1_{}'.format(block, i))
        low_inputs[i] = RCU(rcu_low, n_filters = low_n, name='rb_{}_rcu_l2_{}'.format(block, i))

    fused = MRF(high_input = rcu_high,
                low_inputs = low_inputs,
                n_filters = 256,
                name = 'rb_{}_mrf_'.format(block))
    pooled = CRP(fused, n_filters = 256, name='rb_{}_crp_'.format(block))
    output = RCU(pooled, n_filters = 256, name='rb_{}_rcu_o1_'.format(block))
    return output


def RefineNet(input_shape: tuple, nb_classes: int, variant="single") -> tf.keras.Model:
    """
    Build RefineNet architecture for semantic segmentation based
    on ResNet50
    # Arguments:
    input_shape : tuple
        Shape of input tensor
    nb_classes : int
        Number of classes to segment
    type : str, optional
        Type of returned RefineNet. Supported types: single, 2-cascaded,
        4-cascaded
    # Returns:
        tf.keras.Model of RefineNet
    """
    res = ResNet50(input_shape, refinenet=True)  # build resnet50
    input = res.input  # get input tensor
    res_out = res.output  # get list of outputs
    # set chanel numbers
    res_out[0] = Conv2D(256, 1, padding='same', name='resnet_map1')(res_out[0])
    res_out[1] = Conv2D(256, 1, padding='same', name='resnet_map2')(res_out[1])
    res_out[2] = Conv2D(256, 1, padding='same', name='resnet_map3')(res_out[2])
    res_out[3] = Conv2D(256, 1, padding='same', name='resnet_map4')(res_out[3])
    # perform batch normalization
    for i in range(len(res_out)):
        res_out[i] = BatchNormalization()(res_out[i])

    if variant == "single":  # push all ResNet outs in single RefineBlock
        x = RefineBlock(res_out[0], res_out[1:])
        x = RCU(x, name='rf_rcu_o1_')
        x = RCU(x, name='rf_rcu_o2_')
        x = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(x)
        x = Conv2D(nb_classes, 1, name='rf_pred')(x)
        x = Activation("softmax", name='output')(x)
        return Model(input, x)

    if variant == "2-cascaded":
        # push 2 last ResNet outs in RefineBlock
        x = RefineBlock(res_out[2], res_out[3:])
        # push 2 first ResNet outs in RefineBlock with out of previous RefineBlock
        x = RefineBlock(res_out[0], res_out[1:2] + [x,], block=1)
        x = RCU(x, name='rf_rcu_o1_')
        x = RCU(x, name='rf_rcu_o2_')
        x = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(x)
        x = Conv2D(nb_classes, 1, name='rf_pred')(x)
        x = Activation("softmax", name='output')(x)
        return Model(input, x)

    if variant == "4-cascaded":
        # push last ResNet out in RefineBlock
        x = RefineBlock(res_out[3])
        print("ref1")
        # push next ResNet out in RefineBlock with out of previous RefineBlock
        x = RefineBlock(res_out[2], [x,], block=1)
        print("ref2")
        # push next ResNet out in RefineBlock with out of previous RefineBlock
        x = RefineBlock(res_out[1], [x,], block=2)
        # push next ResNet out in RefineBlock with out of previous RefineBlock
        x = RefineBlock(res_out[0], [x,], block=3)
        x = RCU(x, name='rf_rcu_o1_')
        x = RCU(x, name='rf_rcu_o2_')
        x = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')(x)
        x = Conv2D(nb_classes, 1, name='rf_pred')(x)
        x = Activation("softmax", name='output')(x)
        return Model(input, x)
