from keras.models import Model
from keras.layers import Input, Dropout, Dense, Concatenate, Add, GaussianNoise, Average, Maximum

from keras.optimizers import Adam
from keras import backend as K

from keras.layers import GaussianNoise, Input, Lambda,Subtract, concatenate,Dense, Dropout, Flatten,GlobalAveragePooling2D,Conv2D, Conv1D, Reshape,\
    UpSampling2D, MaxPool2D, MaxPooling1D, GlobalMaxPooling1D, Activation, ZeroPadding2D, MaxPooling2D, MaxPooling3D, Add, Multiply, concatenate, BatchNormalization
import numpy as np
#import utils.kerasutils
#from utils import *
from utils.kerasutils import get_channel_axis
from nets.nets_utils import conv, MaxPoolingND, GlobalAveragePoolingND, dense_layers
from nets.custom_losses import exp_categorical_crossentropy
from nets.inception import InceptionBlock

import keras
import numpy as np
from keras.models import Model
from nets import vgg_bn
from keras import regularizers
import tensorflow as tf
import keras.layers as KL
from keras.layers import *
"""Building a regression network for learning human observer."""

__author__ = 'Gary Y. Li'

def rcnn_block_relu(l, out_num_filters=10, ndims=2, filtersize = 3, trainable_flag=False):
    Conv = getattr(KL, 'Conv%dD' % ndims)
    conv1 = Conv(out_num_filters, filtersize, padding='same', trainable=trainable_flag)
    stack1 = conv1(l)
    stack2 = BatchNormalization()(stack1)
    stack3 = PReLU(trainable=trainable_flag)(stack2)

    conv2 = Conv(out_num_filters, filtersize, padding='same', init='he_normal', trainable=trainable_flag)
    stack4 = conv2(stack3)
    stack5 = Add()([stack1, stack4])
    stack6 = BatchNormalization()(stack5)
    stack7 = PReLU(trainable=trainable_flag)(stack6)

    conv3 = Conv(out_num_filters, filtersize, padding='same', weights=conv2.get_weights(), trainable=trainable_flag)
    stack8 = conv3(stack7)
    stack9 = Add()([stack1, stack8])
    stack10 = BatchNormalization()(stack9)
    stack11 = PReLU(trainable=trainable_flag)(stack10)

    conv4 = Conv(out_num_filters, filtersize, padding='same', weights=conv2.get_weights(), trainable=trainable_flag)
    stack12 = conv4(stack11)
    stack13 = Add()([stack1, stack12])
    stack14 = BatchNormalization()(stack13)
    stack15 = PReLU(trainable=trainable_flag)(stack14)

    return stack15

def recurrentNet_fcm_relu(input_size=(384, 384, 1),ndims=2, num_clus=2, nfilters=16):
    Conv = getattr(KL, 'Conv%dD' % ndims)

    x_in = Input(input_size)
    x = x_in

    for i in range(3):
        x = rcnn_block_relu(x, out_num_filters=nfilters, ndims=ndims, filtersize=3, trainable_flag=True)
    # form deformation field
    # x = rcnn_block(x, out_num_filters=16, ndims=2, filtersize=3)
    x = Conv(filters=24, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv(filters=12, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv(filters=num_clus, kernel_size=3, activation='softmax', padding='same')(x)
    x = Reshape((np.prod((96,96)), num_clus))(x)
    x = Activation('softmax')(x)
    model = Model(inputs=x_in, outputs=x)

    return model


def dense_block(units, dropout=0.2, activation='relu', name='fc1'):
    def layer_wrapper(inp):
        x = Dense(units, name=name)(inp)
        x = BatchNormalization(name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_act'.format(name))(x)
        x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x

    return layer_wrapper

def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1, direction = 'coronal'):
    def layer_wrapper(inp):
        x = Conv2D(units, (3, 3), padding='same', name='block{}_conv{}_{}'.format(block, layer,direction))(inp)
        x = BatchNormalization(name='block{}_bn{}_{}'.format(block, layer,direction))(x)
        x = Activation(activation, name='block{}_act{}_{}'.format(block, layer,direction))(x)
        x = Dropout(dropout, name='block{}_dropout{}_{}'.format(block, layer,direction))(x)
        return x

    return layer_wrapper

def conv_block_onebyonefilter(units, dropout=0.2, activation='relu', block=1, layer=1, direction = 'coronal'):
    def layer_wrapper(inp):
        x = Conv2D(units, (1, 1), padding='same', name='block{}_conv{}_{}'.format(block, layer,direction))(inp)
        x = BatchNormalization(name='block{}_bn{}_{}'.format(block, layer,direction))(x)
        x = Activation(activation, name='block{}_act{}_{}'.format(block, layer,direction))(x)
        x = Dropout(dropout, name='block{}_dropout{}_{}'.format(block, layer,direction))(x)
        return x

    return layer_wrapper


def conv_block_1D(units, dropout=0.2, activation='relu', block=1, layer=1, direction = 'coronal'):
    def layer_wrapper(inp):
        x = Conv1D(units, kernel_size = (3), padding='same', name='block{}_conv{}_{}'.format(block, layer,direction))(inp)
        x = BatchNormalization(name='block{}_bn{}_{}'.format(block, layer,direction))(x)
        x = Activation(activation, name='block{}_act{}_{}'.format(block, layer,direction))(x)
        x = Dropout(dropout, name='block{}_dropout{}_{}'.format(block, layer,direction))(x)
        return x

    return layer_wrapper


def comp_tensor_std(input,direction=1):
    #import tensorflow as tf
    #size = 96
    prj = tf.math.reduce_std(input,direction)
    prj = tf.cast(prj, tf.float32)
    ##print tf.shape(prj)
    return prj


def vertical_pb(input):
    #import tensorflow as tf
    size = 96
    prj = tf.reduce_sum(input,1)
    back_prj = tf.stack([prj] * size, axis = 1)
    return back_prj

def vertical_prj(input):
    #import tensorflow as tf
    size = 96
    prj = tf.reduce_sum(input,1)
    prj = tf.cast(prj, tf.float32)
    ##print tf.shape(prj)
    return prj

def horizontal_prj(input):
    #import tensorflow as tf
    size = 96
    prj = tf.reduce_sum(input,2)
    prj = tf.cast(prj, tf.float32)
    return prj


def horizontal_pb(input):
    #import tensorflow as tf
    size = 96
    prj = tf.reduce_sum(input,2)
    back_prj = tf.stack([prj] * size, axis = 2)
    return back_prj


def total_variation(image):
    noise = tf.image.total_variation(image)
    noise = tf.cast(noise, tf.float32)
    return noise


def abs_and_sum(x):
    abs_x_vector = K.abs(x)
    sum_x = K.sum(abs_x_vector, axis=1)
    return sum_x

def dot_product(tensors):
    return K.sum(tensors[0] * tensors[1],axis=-2,keepdims=True)

def find_max(input):
    max = tf.reduce_max(input)
    return max

def gen_cubic(input):
    #cubic = K.repeat(input,n=96)
    #multiply = tf.constant([96])
    cubic = tf.stack([input] * 96, axis = 3)
    return cubic

def multiply_tensor(tensor_a,tensor_b):
    result = tf.linalg.matmul(tensor_a,tensor_b)
    return result

def transposor_coronal(input):
    transposed = K.permute_dimensions(input,(0,1,3,2,4))
    return transposed


def transposor_transaxial(input):
    transposed = K.permute_dimensions(input,(0,3,2,1,4))
    return transposed


def flipper(input):
    flipped = K.reverse(input,axes=3)
    return flipped




def fineTune_AMO_model7_AddtoGetFms_prjbcktocheckdefXYZ_usingdotproduct_6features_usingMDN_welu_wDO(denseunit_num, dropout_fraction, numMixtures,max_std):
    # shared encoder, each slice gets a probability value from the encoder, add these values from the 3 views and divide by 3 to get the final predicated value
    #encoder.trainable = False
    input_shape = (96, 96, 3)

    input = Input(input_shape)
    input1 = Lambda(lambda input: input[:, :, :, 0:1], output_shape=tuple(input_shape[:-1] + (1,)))(input)
    added_fms_coronal_defect = Reshape((96, 96, 1))(input1)

    input2 = Lambda(lambda input: input[:, :, :, 1:2], output_shape=tuple(input_shape[:-1] + (1,)))(input)
    added_fms_sagittal_defect = Reshape((96, 96, 1))(input2)

    input3 = Lambda(lambda input: input[:, :, :, 2:3], output_shape=tuple(input_shape[:-1] + (1,)))(input)
    added_fms_transaxial_defect = Reshape((96, 96, 1))(input3)



    # defect_status in coronal, saggital and transaxial
    defect_status_coronal_1 = Lambda(lambda x: K.sum(x, axis=1))(added_fms_coronal_defect)
    defect_status_coronal_flat = Lambda(lambda x: K.sum(x, axis=1))(defect_status_coronal_1)
    defect_status_coronal_flat = Lambda(lambda x: K.cast(x, dtype=tf.float32))(defect_status_coronal_flat)


    defect_status_sagittal_1 = Lambda(lambda x: K.sum(x, axis=1))(added_fms_sagittal_defect)
    defect_status_sagittal_flat = Lambda(lambda x: K.sum(x, axis=1))(defect_status_sagittal_1)
    defect_status_sagittal_flat = Lambda(lambda x: K.cast(x, dtype=tf.float32))(defect_status_sagittal_flat)


    defect_status_transaxial_1 = Lambda(lambda x: K.sum(x, axis=1))(added_fms_transaxial_defect)
    defect_status_transaxial_flat = Lambda(lambda x: K.sum(x, axis=1))(defect_status_transaxial_1)
    defect_status_transaxial_flat = Lambda(lambda x: K.cast(x, dtype=tf.float32))(defect_status_transaxial_flat)


    # To confirm x direction sagittal_v and transaxial_v
    sagittal_v = Lambda(vertical_prj)(added_fms_sagittal_defect)
    #sagittal_v = Lambda(lambda x: x + tf.convert_to_tensor(0.001,dtype=tf.float32))(sagittal_v)
    transaxial_v = Lambda(vertical_prj)(added_fms_transaxial_defect)
    #transaxial_v = Lambda(lambda x: x + tf.convert_to_tensor(0.001,dtype=tf.float32))(transaxial_v)
    #x_vector = Subtract()([sagittal_v, transaxial_v])
    difference_x = Lambda(dot_product)([sagittal_v,transaxial_v])
    difference_x = GlobalMaxPooling1D()(difference_x)


    # To confirm y direction coronal_v and transaxial_h
    coronal_v = Lambda(vertical_prj)(added_fms_coronal_defect)
    #coronal_v = Lambda(lambda x: x + tf.convert_to_tensor(0.001,dtype=tf.float32))(coronal_v)

    transaxial_h = Lambda(horizontal_prj)(added_fms_transaxial_defect)
    transaxial_h = Lambda(lambda x:tf.reverse(x,axis=[-2]))(transaxial_h)
    #transaxial_h = Lambda(lambda x: x + tf.convert_to_tensor(0.001,dtype=tf.float32))(transaxial_h)

    #y_vector = Subtract()()
    difference_y = Lambda(dot_product)([transaxial_h, coronal_v])
    difference_y = GlobalMaxPooling1D()(difference_y)

    # To confirm z direction sagittal_h and coronal_h
    sagittal_h = Lambda(horizontal_prj)(added_fms_sagittal_defect)
    #sagittal_h = Lambda(lambda x: x + tf.convert_to_tensor(0.001,dtype=tf.float32))(sagittal_h)

    coronal_h = Lambda(horizontal_prj)(added_fms_coronal_defect)
    #coronal_h = Lambda(lambda x: x + tf.convert_to_tensor(0.001,dtype=tf.float32))(coronal_h)

    #z_vector = Subtract()([sagittal_h, coronal_h])
    difference_z = Lambda(dot_product)([sagittal_h, coronal_h])
    difference_z = GlobalMaxPooling1D()(difference_z)

    #difference_xyz = Concatenate()(
    #    [defect_status_coronal_flat, defect_status_sagittal_flat, defect_status_transaxial_flat, difference_x,difference_y,difference_z])
    difference_xyz = Concatenate()(
        [difference_x,
         difference_y, difference_z])

    difference_xyz = Lambda(lambda x: K.cast(x, dtype=tf.float32))(difference_xyz)
    #difference_xyz = Concatenate()(
    #        [difference_x, difference_y, difference_z])



    x = dense_block(denseunit_num, dropout=0.5, activation='relu', name='fc1')(difference_xyz)
    x = dense_block(denseunit_num, dropout=0.5, activation='relu', name='fc2')(x)
    x = dense_block(denseunit_num, dropout=0.5, activation='relu', name='fc3')(x)
    # x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc4')(x)
    # x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc5')(x)
    # x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc6')(x)
    # x = MixtureDensity(outputDim,numComponents)(x)

    priors_x = dense_block(denseunit_num, dropout=0.5, activation='relu', name='fc5')(x)
    priors_x = dense_block(denseunit_num, dropout=0.5, activation='relu', name='fc6')(priors_x)
    priors_x = dense_block(denseunit_num, dropout=0.5, activation='relu', name='fc7')(priors_x)
    #priors_x = dense_block(size_dense, dropout=0.5, activation='relu', name='fc8')(priors_x)

    priors = Dense(numMixtures, activation='softmax', kernel_initializer='random_normal')(priors_x)
    #priors = Dense(numComponents, activation='softmax', kernel_initializer='random_normal')(x)
    sigmas = Dense(numMixtures, kernel_initializer='random_normal')(x)
    mus = Dense(numMixtures, kernel_initializer='random_normal')(x)
    sigmas = Lambda(lambda x: max_std * K.sigmoid(x))(sigmas)
    mus = Lambda(lambda x: 10 * K.tanh(x))(mus)

    final = concatenate([priors, sigmas, mus])

    net = Model(inputs=input, outputs=final)
    return net
