from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Concatenate, BatchNormalization, \
                                    Conv2DTranspose, UpSampling2D, Activation, Conv2DTranspose, AveragePooling2D, \
                                    DepthwiseConv2D
from tensorflow.keras.models import Model
import tensorflow as tf

import func

########################################################################################################################
# U-Net
def Unet(image_size):
    f = [64, 128, 256, 512, 1024]
    inp = Input((image_size,image_size,3))  # 1

    # contraction path
    c1, p1 = func.contraction(inp, num_filter=f[0])  # c1=480x320x64, p1=240x160x64
    c2, p2 = func.contraction(p1, num_filter=f[1])  # c2=240x160x128, p2=120x80x128
    c3, p3 = func.contraction(p2, num_filter=f[2])  # c3=120x160x128, p3=60x80x256

    # bottle-neck
    bn = func.bottleneck(p3, num_filter=f[3])  # 60x80x512

    # expansion
    e3 = func.expansion(bn, c3, num_filter=f[2]) # c1=480x320x1, p1=240x160x64
    e2 = func.expansion(e3, c2, num_filter=f[1])
    e1 = func.expansion(e2, c1, num_filter=f[0])

    out = Conv2D(2, (3,3), padding='SAME', activation='relu')(e1)
    out = Conv2D(1, (1, 1), padding='SAME', activation='sigmoid')(out)

    model = Model(inp, out)

    return model


########################################################################################################################
# Residual-U-Net
def Res_Unet(image_size):
    f = [64, 128, 256, 512, 1024]
    inp = Input((image_size[0], image_size[1], 1))  # 1

    # contraction path
    c1, p1 = func.residual_contraction(inp, num_filter=f[0])  # 112x112x16
    c2, p2 = func.residual_contraction(p1, num_filter=f[1])  # 56x56x32
    c3, p3 = func.residual_contraction(p2, num_filter=f[2])  # 28x28x64

    # bottle-neck
    bn = func.res_bottleneck(p3, num_filter=f[3])  # 7x7x512

    # expansion
    e3 = func.residual_expansion(bn, c3, num_filter=f[2])
    e2 = func.residual_expansion(e3, c2, num_filter=f[1])
    e1 = func.residual_expansion(e2, c1, num_filter=f[0])

    out = Conv2D(2, (3, 3), padding='SAME', activation='relu')(e1)
    out = Conv2D(1, (1, 1), padding='SAME', activation='sigmoid')(out)

    model = Model(inp, out)

    return model



########################################################################################################################
# Seg-Net
def SegNet(image_size):
    f = [64, 128, 256, 512]
    inp = Input((image_size[0], image_size[1], 1))  # 1

    # contraction path
    c1 = Conv2D(f[0], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(inp)
    c1 = Conv2D(f[0], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c1)
    p1 = MaxPool2D()(c1)

    c2 = Conv2D(f[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(p1)
    c2 = Conv2D(f[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c2)
    p2 = MaxPool2D()(c2)

    c3 = Conv2D(f[2], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(p2)
    c3 = Conv2D(f[2], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c3)
    c3 = Conv2D(f[2], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c3)
    p3 = MaxPool2D()(c3)

    c4 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(p3)
    c4 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c4)
    c4 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c4)
    p4 = MaxPool2D()(c4)

    c5 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(p4)
    c5 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c5)
    c5 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(c5)
    p5 = MaxPool2D()(c5)


    # expansion
    e5 = Conv2DTranspose(filters=f[3],kernel_size=(3,3),strides=(2,2), padding='SAME',activation='relu')(p5)
    e5 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e5)
    e5 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e5)
    e5 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e5)

    e4 = Conv2DTranspose(filters=f[3],kernel_size=(3,3),strides=(2,2), padding='SAME',activation='relu')(e5)
    e4 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e4)
    e4 = Conv2D(f[3], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e4)
    e4 = Conv2D(f[2], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e4)

    e3 = Conv2DTranspose(filters=f[2],kernel_size=(3,3),strides=(2,2), padding='SAME',activation='relu')(e4)
    e3 = Conv2D(f[2], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e3)
    e3 = Conv2D(f[2], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e3)
    e3 = Conv2D(f[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e3)

    e2 = Conv2DTranspose(filters=f[1],kernel_size=(3,3),strides=(2,2), padding='SAME',activation='relu')(e3)
    e2 = Conv2D(f[1], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e2)
    e2 = Conv2D(f[0], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e2)

    e1 = Conv2DTranspose(filters=f[1],kernel_size=(3,3),strides=(2,2), padding='SAME',activation='relu')(e2)
    e1 = Conv2D(f[0], kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')(e1)
    out = Conv2D(1, (1, 1), padding='SAME', activation='sigmoid')(e1)

    model = Model(inp, out)

    return model







########################################################################################################################
# UNet++
def Unetpp(image_size):
    f = [64, 128, 256, 512]

    inp = Input((image_size[0], image_size[1], 1))  # 1

    x00,p00=func.contraction(inp,num_filter=f[0])

    x10,p10=func.contraction(p00,num_filter=f[1])

    x20, p20 = func.contraction(p10, num_filter=f[2])

    x30, p = func.contraction(p20, num_filter=f[3])

    x21 = Conv2DTranspose(filters=f[2],kernel_size=(3,3),strides=(2,2),padding='SAME',activation='relu')(x30)
    x21 = Concatenate()([x20,x21])
    x21, p = func.contraction(x21, num_filter=f[2])

    x11 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x20)
    x11 = Concatenate()([x10, x11])
    x11, p = func.contraction(x11, num_filter=f[1])

    x12 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x21)
    x12 = Concatenate()([x10,x11,x12])
    x12, p = func.contraction(x12, num_filter=f[1])

    x01 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x10)
    x01 = Concatenate()([x00, x01])
    x01, p = func.contraction(x01, num_filter=f[0])

    x02 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x11)
    x02 = Concatenate()([x00,x01,x02])
    x02, p = func.contraction(x02, num_filter=f[0])

    x03 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x12)
    x03 = Concatenate()([x00,x01,x02,x03])
    x03, p = func.contraction(x03, num_filter=f[0])

    out = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='sigmoid')(x03)

    model=Model(inp,out)

    return model



########################################################################################################################
# Residual-UNet++
def Res_Unetpp(image_size):
    f = [64, 128, 256, 512]

    inp = Input((image_size[0], image_size[1], 1))  # 1

    x00,p00=func.residual_contraction(inp,num_filter=f[0])

    x10,p10=func.residual_contraction(p00,num_filter=f[1])

    x20, p20 = func.residual_contraction(p10, num_filter=f[2])

    x30, p = func.residual_contraction(p20, num_filter=f[3])

    x21 = Conv2DTranspose(filters=f[2],kernel_size=(3,3),strides=(2,2),padding='SAME',activation='relu')(x30)
    x21 = Concatenate()([x20,x21])
    x21, p = func.residual_contraction(x21, num_filter=f[2])

    x11 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x20)
    x11 = Concatenate()([x10, x11])
    x11, p = func.residual_contraction(x11, num_filter=f[1])

    x12 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x21)
    x12 = Concatenate()([x10,x11,x12])
    x12, p = func.residual_contraction(x12, num_filter=f[1])

    x01 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x10)
    x01 = Concatenate()([x00, x01])
    x01, p = func.residual_contraction(x01, num_filter=f[0])

    x02 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x11)
    x02 = Concatenate()([x00,x01,x02])
    x02, p = func.residual_contraction(x02, num_filter=f[0])

    x03 = Conv2DTranspose(filters=f[2], kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu')(x12)
    x03 = Concatenate()([x00,x01,x02,x03])
    x03, p = func.residual_contraction(x03, num_filter=f[0])

    out = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='sigmoid')(x03)

    model=Model(inp,out)

    return model



########################################################################################################################
# Dense-UNet (TVTS)
def Dense_Unet(image_shape, blocks, growth_rate, lr, weights=None,
             data_format='channels_last'):
    """ The core of the Dense U-Net.
    # Arguments
        image_shape:  array, ints, shape of the image.
        blocks:       array, ints, the number of dense blocks for each
                      resolution block.
        growth_rate:  float, growth rate (feature maps) at the dense layers.
        lr:           float, learning rate of the optimizer.
        weights:      string, name of the h5 file with the weights (optional)
        data_format:  string to indicate whether channels go first or last.
                      NOTE: The code only works with channels_last for now.
    # Returns the model
    """

    bn_axis = 1 if data_format == 'channels_first' else 3
    comprs = [int(x * growth_rate * 0.50) for x in blocks]

    inputs = Input(shape=image_shape)

    # -------------------------------------------- Dense Block 01
    x1 = func.dense_block(inputs, blocks[0], growth_rate, name='block_1',
                     data_format=data_format)
    x1d = func.down_block(x1, comprs[0], name='down_1',
                     data_format=data_format)

    # -------------------------------------------- Dense Block 02
    x2i = AveragePooling2D(2, padding='same',
                                  data_format=data_format)(inputs)
    x2c = Concatenate(axis=bn_axis)([x2i, x1d])
    x2 = func.dense_block(x2c, blocks[1], growth_rate, name='block_2',
                     data_format=data_format)
    x2d = func.down_block(x2, comprs[1], name='down_2',
                     data_format=data_format)

    # -------------------------------------------- Dense Block 03
    x3i = AveragePooling2D(2, padding='same',
                                  data_format=data_format)(x2i)
    x3c = Concatenate(axis=bn_axis)([x3i, x2d])
    x3 = func.dense_block(x3c, blocks[2], growth_rate, name='block_3',
                     data_format=data_format)
    x3d = func.down_block(x3, comprs[2], name='down_3',
                     data_format=data_format)

    # -------------------------------------------- Dense Block 04
    x4i = AveragePooling2D(2, padding='same',
                                  data_format=data_format)(x3i)
    x4c = Concatenate(axis=bn_axis)([x4i, x3d])
    x4 = func.dense_block(x4c, blocks[3], growth_rate, name='block_4',
                     data_format=data_format)
    up4 = func.upsampling_block(x4, comprs[3], name='up_4',
                           data_format=data_format)

    # -------------------------------------------- Dense Block 05
    x5 = Concatenate(axis=bn_axis)([x3, up4])
    x5 = func.dense_block(x5, blocks[4], growth_rate, name='block_5',
                     data_format=data_format)
    up5 = func.upsampling_block(x5, comprs[4], name='up_5',
                           data_format=data_format)

    # -------------------------------------------- Dense Block 06
    x6 = Concatenate(axis=bn_axis)([x2, up5])
    x6 = func.dense_block(x6, blocks[5], growth_rate, name='block_6',
                     data_format=data_format)
    up6 = func.upsampling_block(x6, comprs[5], name='up_6',
                           data_format=data_format)

    # -------------------------------------------- Dense Block 07
    x7 = Concatenate(axis=bn_axis)([x1, up6])
    x7 = func.dense_block(x7, blocks[6], growth_rate, name='block_7',
                     data_format=data_format)
    x7 = Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                       data_format=data_format)(x7)

    # -------------------------------------------- RESHAPE BLOCK
    # This part is not yet prepared for 'channels_first'
    # x8 = layers.Permute((3,1,2))(x7)
    # x8 = layers.Reshape((2, int(image_shape[0]*image_shape[1])))(x8)
    # x8 = layers.Permute((2,1))(x8)
    # x8 = layers.Activation('softmax')(x8)

    # -------------------------------------------- PARAMETERS
    model = Model(inputs=inputs, outputs=x7)


    return model




########################################################################################################################
# Mobile-CellNet
def mobile_polypNet(img_dim, pre_trained=False):
    if pre_trained:
        mobileNetv2 = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        mobileNetv2.trainable = False
        mobileNetv2 = Mobile - UNet.ipynb
        inp = mobileNetv2.input
        x = mobileNetv2.get_layer('expanded_conv_project_BN').output
        c1 = mobileNetv2.get_layer('block_2_add').output
        c2 = mobileNetv2.get_layer('block_5_add').output
        c3 = mobileNetv2.get_layer('block_12_add').output
        c4 = mobileNetv2.get_layer('out_relu').output
        bn = bottleneck(c4, 1280, 320, add=False)
        bn = bottleneck(bn, 960, 160, add=False)
        bn = bottleneck(bn, 960, 160, add=False)
        bn = bottleneck(bn, 576, 96, add=False)

    else:
        ## self defined MobileNetV2
        inp = Input(shape=(img_dim, img_dim, 3))
        x = Conv2D(32, (3, 3), strides=1, activation=None, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=1, activation=None, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)
        x = Conv2D(32, (1, 1), strides=1, activation=None, padding='same')(x)
        x = BatchNormalization()(x)

        # first bottleneck layer
        c1 = func.mobile_bottleneck(x, 48, 8, strides=2)
        c1 = func.mobile_bottleneck(c1, 48, 8, strides=1)
        c1 = func.mobile_bottleneck(c1, 48, 8, strides=1)

        # second bottleneck layer
        c2 = func.mobile_bottleneck(c1, 96, 16, strides=2)
        c2 = func.mobile_bottleneck(c2, 96, 16, strides=1)
        c2 = func.mobile_bottleneck(c2, 96, 16, strides=1)

        c3 = func.mobile_bottleneck(c2, 144, 32, strides=2)
        c3 = func.mobile_bottleneck(c3, 144, 32, strides=1)
        c3 = func.mobile_bottleneck(c3, 144, 32, strides=1)

        c4 = func.mobile_bottleneck(c3, 144, 32, strides=2)
        c4 = func.mobile_bottleneck(c4, 144, 32, strides=1)
        c4 = func.mobile_bottleneck(c4, 144, 32, strides=1)

        c5 = func.mobile_bottleneck(c4, 144, 32, strides=2)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)

    e4 = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same')(c5)
    e4 = Concatenate()([e4, c4])
    e4 = func.mobile_bottleneck(e4, 144, 32, add=False)
    e4 = func.mobile_bottleneck(e4, 144, 32, strides=1)

    e3 = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same')(e4)
    e3 = Concatenate()([e3, c3])
    e3 = func.mobile_bottleneck(e3, 96, 16, add=False)
    e3 = func.mobile_bottleneck(e3, 96, 16, strides=1)

    # upsampling 2
    e2 = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same')(e3)
    e2 = Concatenate()([e2, c2])
    e2 = func.mobile_bottleneck(e2, 96, 16, add=False)
    e2 = func.mobile_bottleneck(e2, 96, 16, strides=1)

    # e2=Conv2D(32,kernel_size=3,strides=1,padding='same')(e2)

    e1 = Conv2DTranspose(8, kernel_size=3, strides=2, padding='same')(e2)
    e1 = Concatenate()([e1, c1])
    e1 = func.mobile_bottleneck(e1, 48, 8, add=False)
    e1 = func.mobile_bottleneck(e1, 48, 8, strides=1)

    # upsampling 1
    e0 = Conv2DTranspose(8, kernel_size=3, strides=2, padding='same')(e1)
    e0 = Concatenate()([e0, x])
    e0 = func.mobile_bottleneck(e0, 48, 8, add=False)
    e0 = func.mobile_bottleneck(e0, 48, 8, strides=1)

    out = Conv2D(32, 3, strides=1, padding='same')(e0)
    out = BatchNormalization()(out)
    out = Activation(tf.nn.relu6)(out)
    out = Conv2D(16, 3, strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation(tf.nn.relu6)(out)

    out = Conv2D(1, 3, strides=1, padding='same', activation='sigmoid')(out)

    model = Model(inp, out)
    return model






def mobile_polypNet_maxPool(img_size, pre_trained=False):
    if pre_trained:
        mobileNetv2 = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        mobileNetv2.trainable = False
        mobileNetv2 = Mobile - UNet.ipynb
        inp = mobileNetv2.input
        x = mobileNetv2.get_layer('expanded_conv_project_BN').output
        c1 = mobileNetv2.get_layer('block_2_add').output
        c2 = mobileNetv2.get_layer('block_5_add').output
        c3 = mobileNetv2.get_layer('block_12_add').output
        c4 = mobileNetv2.get_layer('out_relu').output
        bn = func.mobile_bottleneck(c4, 1280, 320, add=False)
        bn = func.mobile_bottleneck(bn, 960, 160, add=False)
        bn = func.mobile_bottleneck(bn, 960, 160, add=False)
        bn = func.mobile_bottleneck(bn, 576, 96, add=False)


    else:

        ## self defined MobileNetV2
        inp = Input(shape=(img_size, img_size, 3))
        x = Conv2D(32, (3, 3), strides=1, activation=None, padding='same')(inp)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=1, activation=None, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.relu6)(x)
        x = Conv2D(32, (1, 1), strides=1, activation=None, padding='same')(x)
        x = BatchNormalization()(x)

        # first bottleneck layer
        c1 = MaxPool2D()(x)
        c1 = func.mobile_bottleneck(c1, 48, 8, strides=1, add=False)
        c1 = func.mobile_bottleneck(c1, 48, 8, strides=1)
        c1 = func.mobile_bottleneck(c1, 48, 8, strides=1)

        # second bottleneck layer
        c2 = MaxPool2D()(c1)
        c2 = func.mobile_bottleneck(c2, 96, 16, strides=1, add=False)
        c2 = func.mobile_bottleneck(c2, 96, 16, strides=1)
        c2 = func.mobile_bottleneck(c2, 96, 16, strides=1)

        c3 = MaxPool2D()(c2)
        c3 = func.mobile_bottleneck(c3, 144, 32, strides=1, add=False)
        c3 = func.mobile_bottleneck(c3, 144, 32, strides=1)
        c3 = func.mobile_bottleneck(c3, 144, 32, strides=1)

        c4 = MaxPool2D()(c3)
        c4 = func.mobile_bottleneck(c4, 144, 32, strides=1, add=False)
        c4 = func.mobile_bottleneck(c4, 144, 32, strides=1)
        c4 = func.mobile_bottleneck(c4, 144, 32, strides=1)

        c5 = MaxPool2D()(c4)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1, add=False)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)
        c5 = func.mobile_bottleneck(c5, 144, 32, strides=1)

    e4 = UpSampling2D()(c5)
    e4 = Concatenate()([e4, c4])
    e4 = func.mobile_bottleneck(e4, 144, 32, add=False)
    e4 = func.mobile_bottleneck(e4, 144, 32, strides=1)

    e3 = UpSampling2D()(e4)
    e3 = Concatenate()([e3, c3])
    e3 = func.mobile_bottleneck(e3, 96, 16, add=False)
    e3 = func.mobile_bottleneck(e3, 96, 16, strides=1)

    # upsampling 2
    e2 = UpSampling2D()(e3)
    e2 = Concatenate()([e2, c2])
    e2 = func.mobile_bottleneck(e2, 96, 16, add=False)
    e2 = func.mobile_bottleneck(e2, 96, 16, strides=1)

    # e2=Conv2D(32,kernel_size=3,strides=1,padding='same')(e2)
    e1 = UpSampling2D()(e2)
    e1 = Concatenate()([e1, c1])
    e1 = func.mobile_bottleneck(e1, 48, 8, add=False)
    e1 = func.mobile_bottleneck(e1, 48, 8, strides=1)

    # upsampling 1
    e0 = UpSampling2D()(e1)
    e0 = Concatenate()([e0, x])
    e0 = func.mobile_bottleneck(e0, 48, 8, add=False)
    e0 = func.mobile_bottleneck(e0, 48, 8, strides=1)

    out = Conv2D(32, 3, strides=1, padding='same')(e0)
    out = BatchNormalization()(out)
    out = Activation(tf.nn.relu6)(out)
    out = Conv2D(16, 3, strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation(tf.nn.relu6)(out)
    out = Conv2D(1, 3, strides=1, padding='same', activation='sigmoid')(out)
    model = Model(inp, out)

    return model