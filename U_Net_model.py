# Import all modules for model building
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D,Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam, SGD


def DownBlock(inputs, n_filters):
    x = inputs
    y1 = Conv2D(n_filters, 3, padding='same')(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    
    y2 = Conv2D(n_filters, 3, padding='same')(concatenate([x, y1]))
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    
    z = Conv2D(n_filters, 1, padding='same')(concatenate([x, y1, y2]))
    z = BatchNormalization()(z)
    z = ReLU()(z)
    
    next_layer = MaxPooling2D((2, 2))(z)
    return next_layer, z


def UpBlock(prev_layer_input, skip_layer_input, n_filters_transpose, n_filters):
    x1 = prev_layer_input
    x2 = skip_layer_input
    y1 = Conv2DTranspose(n_filters_transpose, (3,3), strides=(2,2), padding='same')(x1)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y2 = Conv2D(n_filters, 1, padding='same')(concatenate([y1, x2]))
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y3 = Conv2D(n_filters, 3, padding='same')(y2)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y4 = Conv2D(n_filters, 3, padding='same')(concatenate([y2, y3]))
    y4 = BatchNormalization()(y4)
    y4 = ReLU()(y4)
    next_layer = Conv2D(n_filters, 1, padding='same')(concatenate([y2, y3, y4]))
    next_layer = BatchNormalization()(next_layer)
    next_layer = ReLU()(next_layer)
    return next_layer
  
  
def DenseUnet(input_shape, n_classes, n_filters=64):
    inputs = Input(input_shape)
    
    cblock1 = DownBlock(inputs, n_filters)
    cblock2 = DownBlock(cblock1[0],n_filters*2)
    cblock3 = DownBlock(cblock2[0], n_filters*4)
    cblock4 = DownBlock(cblock3[0], n_filters*8)
    cblock5 = DownBlock(cblock4[0], n_filters*8) 
    
    ublock6 = UpBlock(cblock5[0], cblock5[1],  n_filters * 8, n_filters * 8)
    ublock7 = UpBlock(ublock6, cblock4[1],  n_filters * 8, n_filters * 4)
    ublock8 = UpBlock(ublock7, cblock3[1],  n_filters * 4, n_filters * 2)
    ublock9 = UpBlock(ublock8, cblock2[1],  n_filters * 2, n_filters)
    ublock10 = UpBlock(ublock9, cblock1[1],  n_filters, n_filters)

    conv10 = Conv2D(n_classes, 3, padding='same', activation="softmax")(ublock10)
    
    # Define the model
    model = Model(inputs=inputs, outputs=conv10)

    return model  
