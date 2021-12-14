import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Concatenate, BatchNormalization, \
                                    Conv2DTranspose, UpSampling2D, Add, Activation, Conv2DTranspose, AveragePooling2D, \
                                    Dropout, DepthwiseConv2D

import numpy as np
import cv2 as cv

from skimage.io import imread_collection

import  os

########################################################################################################################
# Image loader function
def load_data(path, flag=0):
    x = imread_collection(os.path.join(path, '*.tiff'))
    if flag == 1:
        data = np.zeros([len(x.files), 224, 224, 3], dtype='uint8')
    else:
        data = np.zeros([len(x.files), 224, 224], dtype='uint8')
    for i, fname in enumerate(x.files):
        data[i] = cv.imread(fname, flags=flag)

    return data

########################################################################################################################
# Building blocks
def contraction(x,num_filter,kernel_size=(3,3),pool_size=(2,2),strides=1,padding='SAME'):
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(x)
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(c)
    p=MaxPool2D(pool_size)(c)
    return c,p

def expansion(x,x_skip,num_filter,kernel_size=(3,3),pool_size=(2,2),strides=1,padding='SAME'):
    us=Conv2DTranspose(filters=num_filter,kernel_size=kernel_size,strides=(2,2), padding=padding,activation='relu')(x)
    conc=Concatenate()([us,x_skip])
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(conc)
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(c)
    return c

def bottleneck(x,num_filter,kernel_size=(3,3),pool_size=(2,2),strides=1,padding='SAME'):
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(x)
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(c)
    return c

def residual_conv(x,num_filter,kernel_size=(3,3),strides=1,padding='SAME'):
    c1=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(x)
    c2=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation=None)(c1)
    skip=Add()([c1,c2])
    c=Activation('relu')(skip)
    return c


def residual_contraction(x,num_filter,kernel_size=(3,3),pool_size=(2,2),strides=1,padding='SAME'):
    c=residual_conv(x,num_filter)
    p=MaxPool2D(pool_size)(c)
    return c,p


def residual_expansion(x,x_skip,num_filter,kernel_size=(3,3),pool_size=(2,2),strides=1,padding='SAME'):
    us = Conv2DTranspose(filters=num_filter, kernel_size=kernel_size, strides=(2, 2), padding=padding,
                         activation='relu')(x)
    conc=Concatenate()([us,x_skip])
    c=residual_conv(conc,num_filter,kernel_size=kernel_size,strides=strides,padding=padding)
    return c

def res_bottleneck(x,num_filter,kernel_size=(3,3),pool_size=(2,2),strides=1,padding='SAME'):
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(x)
    c=Conv2D(num_filter,kernel_size=kernel_size,strides=strides,padding=padding,activation='relu')(c)
    return c


def dense_block(x, blocks, growth_rate, name, data_format='channels_last'):
    """A dense block.
    # Arguments
        x:          input tensor.
        blocks:     integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers (output maps).
        name:       string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
      x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1),
                     data_format=data_format)
    return x



def conv_block(x, growth_rate, name, data_format='channels_last'):
    """ A building block for a dense block.
    # Arguments
        x:          input tensor.
        growth_rate: float, growth rate at dense layers.
        name:       string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 1 if data_format=='channels_first' else 3
    x1 = Conv2D(growth_rate, 4, activation=None, padding='same',
                       use_bias=False, data_format=data_format,
                       name=name + '_conv')(x)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   renorm=True, name=name + '_bn')(x1)
    x1 = Activation('elu')(x1)
    x1 = Dropout(rate=0.2, name=name + '_drop')(x1)
    x  = Concatenate(axis=bn_axis, name=name + '_conc')([x,x1])
    return x



def down_block(x, out_channels, name, data_format='channels_last'):
    """ The downsampling block, which contains a feature reduction layer,
    (Conv1x1) + BRN + ELU, and a average pooling layer.
    # Arguments
        x:            input tensor.
        out_channels: float, number of output channels.
        name:         string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 1 if data_format=='channels_first' else 3
    x = Conv2D(out_channels, 1, activation=None, padding='same',
                      use_bias=False, data_format=data_format,
                      name=name + '_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  renorm=True, name=name + '_bn')(x)
    x = Activation('elu')(x)
    x = AveragePooling2D(2, padding='same',
                                data_format=data_format)(x)
    return x


def upsampling_block(x, out_channels, name, data_format='channels_last'):
    """ The upsampling block, which contains a transpose convolution.
    # Arguments
        x:            input tensor.
        out_channels: float, number of output channels.
        name:         string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 1 if data_format=='channels_first' else 3
    x = Conv2DTranspose(out_channels, 4, strides=(2, 2),
                               activation=None, padding='same',
                               use_bias=False, data_format=data_format,
                               name=name + '_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  renorm=True, name=name + '_bn')(x)
    x = Activation('elu')(x)
    return x


########################################################################################################################
# Loss functions
def dice_coef(y_true, y_pred, smooth=0):
    y_true=tf.cast(y_true, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def jaccard_loss(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return (1 - jac)

def mixed_loss(y_true,y_pred):
    dice_loss=dice_coef_loss(y_true,y_pred)
    jac_loss=jaccard_loss(y_true,y_pred)
    loss=(dice_loss*0.5)+(jac_loss*0.5)
    return loss

########################################################################################################################
# Performance functions
def model_eval(model, X_train, y_train, X_test, y_test):
    print('evaluating training images...')
    for i in range(X_train.shape[0]):
        y_pred=model.predict(X_train)
        dc=dice_coef(y_train[i,:445,:303],y_pred[:445,:303])
    #train_loss, train_dice, train_acc = model.evaluate(X_train, y_train, verbose=False)
    #test_loss, test_dice, test_acc = model.evaluate(X_test, y_test, verbose=False)
    #print("===" * 35)
    #print("Model training performance:\n\t\t\t Loss:{}\t Accuracy:{}\t Dice:{}".format(np.round(train_loss, 4),
    #                                                                                   np.round(train_acc, 4),
    #                                                                                   np.round(train_dice, 4)))
    #print("===" * 35)
    #print("Model testing performance:\n\t\t\t Loss:{}\t Accuracy:{}\t Dice:{}".format(np.round(test_loss, 4),
     ##                                                                                 np.round(test_acc, 4),
      #                                                                                np.round(test_dice, 4)))
    #print("===" * 35)
    #pred_train = model.predict(X_train)
    #pred_test = model.predict(X_test)
    print(i,dc)



def mobile_conv(x, num_filters, strides=1):
    x = DepthwiseConv2D(3, strides=strides, padding='SAME')(x)  # DW-Conv
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(num_filters, strides=1, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def conv(x, num_filters, strides=1, relu=True):
    x = Conv2D(num_filters, kernel_size=(3, 3), strides=strides, padding='SAME')(x)
    x = BatchNormalization()(x)
    if relu:
        x = Activation(tf.nn.relu6)(x)
    return x


def expansion_mobile_conv(x, x_skip, num_filters):
    x = UpSampling2D()(x)
    x = Concatenate()([x, x_skip])
    x = mobile_conv(x, num_filters=num_filters)
    return x


def mobile_bottleneck(x, expand_filters, contract_filters, strides=1, add=True):
    c = Conv2D(expand_filters, (1, 1), strides=1, activation=None, padding='same')(x)
    c = BatchNormalization()(c)
    c = Activation(tf.nn.relu6)(c)
    # c=Activation('relu')(c)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', activation=None)(c)
    c = BatchNormalization()(c)
    c = Activation(tf.nn.relu6)(c)
    # c=Activation('relu')(c)
    c = Conv2D(contract_filters, (1, 1), strides=1, padding='same', activation=None)(c)
    c = BatchNormalization()(c)
    if (strides == 1) and (add):
        c = Add()([x, c])
    return c


########################################################################################################################
# Model Evaluation
def performance_measures(y_true, y_pred, disp=False):
    tp = np.sum(np.logical_and((y_true.astype(np.uint8) == 1), (y_pred.astype(np.uint8) == 1)))
    tn = np.sum(np.logical_and((y_true.astype(np.uint8) == 0), (y_pred.astype(np.uint8) == 0)))
    fp = np.sum(np.logical_and((y_true.astype(np.uint8) == 0), (y_pred.astype(np.uint8) == 1)))
    fn = np.sum(np.logical_and((y_true.astype(np.uint8) == 1), (y_pred.astype(np.uint8) == 0)))
    sen = tp / (tp + fn)  # recall
    spc = tn / (tn + fp)
    prec = tp / (tp + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    dice = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    jacc = tp / (tp + fp + fn)
    if sen == 0:
        f2 = 0
    else:
        f2 = (5 * prec * sen) / ((4 * prec) + sen)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if disp:
        print('Sensitivity: {}'.format(np.round(sen, 4)))
        print('Specificity: {}'.format(np.round(spc, 4)))
        print('Accuracy: {}'.format(np.round(acc, 4)))
        print('F2 Score: {}'.format(np.round(f2, 4)))
        print('Dice Score: {}'.format(np.round(dice, 4)))
        print('Jaccard Score: {}'.format(np.round(jacc, 4)))
        print('MCC: {}'.format(np.round(mcc, 4)))

    return sen, spc, acc, f2, mcc, dice, jacc, iou


def test_performance(model, X_test, y_test, disp=False):
    sensitivity = 0
    specificity = 0
    accuracy = 0
    f2_score = 0
    dice_2 = 0
    dice_3 = 0
    iou_2 = 0
    jac_coeff = 0

    params = np.round(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])/10**6,2) + np.round(np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights]) / 10 ** 6, 2)
    num = X_test.shape[0]
    _, _, auc, dice, mae, iou = model.evaluate(X_test, y_test, verbose=False)

    for i in range(num):
        idx = i
        pred = model.predict(X_test[np.newaxis, idx, ...])[0, ..., 0]
        _, pred = cv.threshold(pred, 0.5, 1, cv.THRESH_BINARY)

        test_y = y_test[idx]

        sen, spc, acc, f2, _, dc, jac, io_2 = performance_measures(test_y, pred)

        dc_2 = 2 * io_2 / (1 + io_2)

        sensitivity = sensitivity + sen
        specificity = specificity + spc
        accuracy = accuracy + acc
        f2_score = f2_score + f2
        dice_2 = dice_2 + dc
        iou_2 = iou_2 + io_2
        jac_coeff = jac_coeff + jac
        dice_3 = dice_3 + dc_2

        if disp:
            print('Sensitivity: ', np.round(sensitivity / num, 4))
            print('Specificity: ', np.round(specificity / num, 4))
            print('Accuracy: ', np.round(accuracy / num, 4))
            print('AUC: ', np.round(auc / num, 4))
            print('Dice coeff: ', np.round(dice / num, 4))
            print('Dice 2 coeff: ', np.round(dice_2 / num, 4))
            print('Jaccard coeff: ', np.round(jac_coeff / num, 4))
            print('F2 score: ', np.round(f2_score / num, 4))
            print('MAE: ', np.round(mae / num, 4))
            print('meanIoU: ', np.round(iou / num, 4))
            print('meanIoU_2: ', np.round(iou_2 / num, 4))

    return params, np.round(sensitivity / num, 4), np.round(specificity / num, 4), np.round(accuracy / num, 4), \
           np.round(f2_score / num, 4), np.round(dice_2 / num, 4), np.round(iou_2 / num, 4), np.round(jac_coeff / num, 4), \
           np.round(auc, 4), np.round(dice, 4), np.round(mae, 4), np.round(iou, 4), np.round(dice_3 / num, 4)

