"""
#Trains a ResNet on the CIFAR10 dataset.
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
ResNet v2:
[Identity Mappings in Deep Residual Networks
](https://arxiv.org/pdf/1603.05027.pdf)
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v1|  3| 92.16 %|                 91.25 %|35
ResNet32   v1|  5| 92.46 %|                 92.49 %|50
ResNet44   v1|  7| 92.50 %|                 92.83 %|70
ResNet56   v1|  9| 92.71 %|                 93.03 %|90
ResNet110  v1| 18| 92.65 %|            93.39+-.16 %|165
ResNet164  v1| 27|     - %|                 94.07 %|  -
ResNet1001 v1|N/A|     - %|                 92.39 %|  -
&nbsp;
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v2|  2|     - %|                     - %|---
ResNet32   v2|N/A| NA    %|            NA         %| NA
ResNet44   v2|N/A| NA    %|            NA         %| NA
ResNet56   v2|  6| 93.01 %|            NA         %|100
ResNet110  v2| 12| 93.15 %|            93.63      %|180
ResNet164  v2| 18|     - %|            94.54      %|  -
ResNet1001 v2|111|     - %|            95.08+-.14 %|  -
"""

import keras
import numpy as np
import keras.backend as K
#from tensorflow.keras import backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Activation, BatchNormalization
from keras.layers import AveragePooling2D, Input, Flatten
from resnet import cifar100_resnet

def get_model(dataset='mnist', input_tensor=None, input_shape=None, num_classes=10):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar-10' or 'cifar-100') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    input_shape: optional shape tuple
    :return: The model; a Keras 'Model' instance.
    """
    assert dataset in ['mnist','fashion_mnist', 'svhn', 'cifar-10', 'cifar-100','celeb'], \
        "dataset parameter must be either 'mnist', 'svhn', 'cifar-10' or 'cifar-100'"

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_shape):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if dataset == 'mnist' or dataset == 'fashion_mnist':
        # ## LeNet-5
        x = Conv2D(32, (3, 3), padding='same', kernel_initializer="he_normal", name='conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten()(x)

        x = Dense(128, kernel_initializer="he_normal", name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)
        # x = Dropout(0.2)(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)

    elif dataset == 'svhn':
        # ## LeNet-5
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        x = Flatten()(x)

        x = Dense(512, kernel_initializer='he_normal', name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(128, kernel_initializer="he_normal", name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)
        # x = Dropout(0.2)(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x)

    elif dataset == 'cifar-10' or dataset == 'celeb':
        # Block 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer="he_normal", name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer="he_normal", name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(196, (3, 3), padding='same', kernel_initializer="he_normal", name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)

        x = Dense(256, kernel_initializer="he_normal", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='lid')(x)

        x = Dense(num_classes, kernel_initializer="he_normal")(x)
        x = Activation('softmax')(x)

        # Create model.
        model = Model(img_input, x)


    elif dataset == "cifar-100":
        model = cifar100_resnet(depth=7, num_classes = num_classes)


    return model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
