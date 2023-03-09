# Import all modules for model building
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Add


def VGG16(input_shape):
    """
    Build VGG16 network with input_shape
    """
    inp = Input(shape=input_shape)
    # first conv block with maxpooling
    x = Conv2D(64, 3, padding="same", activation="relu")(inp)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    # second conv block with maxpooling
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D()(x) 
    # third conv block with maxpooling
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(name="maxpool3")(x)
    # forth conv block with maxpooling
    x = Conv2D(512, 3, padding="same", activation="relu")(x)
    x = Conv2D(512, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(name="maxpool4")(x)
    # fifth conv block with maxpooling
    x = Conv2D(512, 3, padding="same", activation="relu")(x)
    x = Conv2D(512, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(name="pooling_out")(x)
    return Model(inputs=inp, outputs=x)
    
    
def FCN_32x(input_shape, n_classes):
    vgg = VGG16(input_shape)
    x = Conv2D(n_classes, 1, padding="same", name="vgg_out")(vgg.get_layer("pooling_out").output)
    x = Conv2DTranspose(n_classes, 64, 32, activation="softmax",
                        padding="same", name="fcn_32")(x)
    x = Model(inputs=vgg.input, outputs=x)
    return x
    
    
def FCN_16x(input_shape, n_classes):
    fcn_32 = FCN_32x(input_shape, n_classes)
    x = Conv2DTranspose(n_classes, 4, 2, 
                        padding="same")(fcn_32.get_layer("vgg_out").output)
    y = Conv2D(n_classes, 1, padding="same")(fcn_32.get_layer("maxpool4").output)
    x = Add(name="skip")([x, y])
    x = Conv2DTranspose(n_classes, 32, 16, activation="softmax",
                        padding="same", name="fcn_16")(x)
    x = Model(inputs=fcn_32.input, outputs=x)
    return x


def FCN_8x(input_shape, n_classes):
    fcn_16 = FCN_16x(input_shape, n_classes)
    x = Conv2DTranspose(n_classes, 4, 2, 
                        padding="same")(fcn_16.get_layer("skip").output)
    y = Conv2D(n_classes, 1, padding="same")(fcn_16.get_layer("maxpool3").output)
    x = Add()([x, y])
    x = Conv2DTranspose(n_classes, 16, 8, activation="softmax",
                        padding="same", name="fcn_8")(x)
    x = Model(inputs=fcn_16.input, outputs=x)
    return x