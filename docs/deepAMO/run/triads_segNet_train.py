import os
import sys
sys.path.append('/netscratch/garyli2/anaconda2/lib/python2.7/site-packages')
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import pydot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
sys.path.append('/netscratch/garyli2/cnn_ho/')
from utils.im_reading import load_4Dimage_from_folder
from nets.unet import build_net
from nets.custom_losses import (exp_dice_loss, exp_categorical_crossentropy, combine_loss)
from utils.segmentation_training import segmentation_training
from utils.kerasutils import get_image, correct_data_format, save_model_summary, get_channel_axis
from utils.imageutils import map_label
from utils.input_data import InputSegmentationArrays

if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

input_path = '/netscratch/garyli2/cnn_ho_images/pilot_human_observer/triads_seg_images/'
output_path = ('/netscratch/garyli2/cnn_ho/experiments/outputs/'
              'triads_segmentation_pilotHOS/')
output_image_path = output_path + '/images/'
output_model_path = output_path + '/model/'

#
if output_path is not None and not os.path.exists(output_path):
    os.makedirs(output_path)

this_file = '/netscratch/garyli2/cnn_ho/run/triads_segNet_train.py'
shutil.copy(this_file, output_path)

train_portion = 0.8
valid_portion = 0.2

# Load data
image_array, label_array = load_4Dimage_from_folder(input_path, (96,96), HE=False,Truc=False,Aug=False)
print "image_array, label_array genereation done"
# plt.figure(figsize=(6, 3))
# ax = plt.subplot(1, 2, 1)
# ax.imshow(image_array[59], cmap=cm.Greys_r, interpolation='none')
# ax.axis('off')
#
# ax = plt.subplot(1, 2, 2)
# ax.imshow(label_array[59], interpolation='none')
# ax.axis('off')
# plt.show()

image_train = image_array[0:int(train_portion*len(image_array)),:,:,:]
label_train = label_array[0:int(train_portion*len(image_array)),:,:]
image_valid = image_array[int(train_portion*len(image_array)):len(image_array),:,:,:]
label_valid = label_array[int(train_portion*len(image_array)):len(image_array),:,:]

#this gives all the class label values present in the train label data
unique_labels = np.unique(label_valid)
print 'unique_labels: '
print unique_labels
#
# # Correct data format
#image_train = np.expand_dims(image_train, axis=3)
label_train = np.expand_dims(label_train, axis=3)
#image_valid = np.expand_dims(image_valid, axis=3)
label_valid = np.expand_dims(label_valid, axis=3)
#
input_arrays = InputSegmentationArrays(image_train, label_train, image_valid, label_valid)
# # Model arguments
model_args = dict(
    num_classes=input_arrays.get_num_classes(),
    base_num_filters=16,
    image_size=(96, 96),
    dropout_rate=0.5,
    optimizer=Adam(lr=1e-3),
    conv_order='conv_first',
    kernel_size=3,
    activation='relu',
    net_depth=3,
    convs_per_depth=2,
    noise_std=1.0,
    loss=combine_loss([exp_dice_loss(exp=1), exp_categorical_crossentropy(exp=1)], [0.8, 0.2]),
    #loss = exp_dice_loss(exp=1.0)
)

# # Training arguments
train_args = dict(
    input_arrays=input_arrays,
    output_path=output_path,
    output_model_path=output_model_path,
    output_image_path=output_image_path,
    batch_size=200,
    n_epochs=200,
    print_freq=0.0,
    save_image_batches=1,
    use_lumped_dice=False,
    model_choose_portion=0.3,
)

# Image augmentation
use_aug = False
if use_aug:
    aug_args = dict(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=10.0,
        shift_range=[0.1, 0.1],
        zoom_range=[0.9, 1.1],
        channel_shift_range=0.,
        fill_mode='constant',
        cval=0.,
        flip=None,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        augmentation_probability=1.0,
        ndim=4  # Input array size. 4 for 2D image and 5 for 3D image
    )
else:
    aug_args = None
#
# ####### Manual inputs end here #######


print 'image_train.shape ', image_train.shape
print 'image_valid.shape ', image_valid.shape
print 'label_train.shape ', label_train.shape
# for label in range(input_arrays.get_num_classes()):
#     print 'Sum of pixels with value of %d %d' % (label, (label_train == label).sum())
# print 'Loaded data successfully!'

# idx = np.random.randint(len(input_arrays.image_train))
# img = input_arrays.image_train[idx]
# lab = input_arrays.label_train[idx]

# # Display an image and its segmentation for sanity check
# plt.figure(figsize=(6, 3))
# ax = plt.subplot(1, 2, 1)
# ax.imshow(img, cmap=cm.Greys_r, interpolation='none')
# ax.axis('off')
#
# ax = plt.subplot(1, 2, 2)
# ax.imshow(map_label(lab), interpolation='none')
# ax.axis('off')
# plt.show()
# print 'Min value on image', np.min(img)
# print 'Max value on image', np.max(img)
# print 'Maximum class value on label image', np.max(lab)
#
#
net = build_net(**model_args)
segmentation_model = Model(inputs=net.input, outputs=net.get_layer('activation_8').output)

def softargmax(x, beta=1e2):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

def softargmax_np(x, beta=1e2):
  #x = tf.convert_to_tensor(x)
  x_range = np.range(x.shape.as_list()[-1], dtype=x.dtype)
  return np.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

import math

# from PIL import Image

# def sigmoid(x):
#     x=np.reshape(x, (9216,1))
#     x = 1 / (1 + math.exp(-x))
#     x= np.reshape(x, (96,96))
#     return x

def get_segmentation(image):
    #x= segmentation_model.predict(image)
    #return x.argmax(get_channel_axis())
    #return segmentation_model.predict(image).argmax(get_channel_axis())
    return segmentation_model.predict(image)[:,:,:,1]

# Save model summary
#plot_model(net, show_shapes=True, show_layer_names=True, to_file=output_path + '/net.pdf')
save_model_summary(net, output_path + '/net_summary.txt')
print net.summary()
#
# In [6]:

segmentation_training(
    net=net,
    get_segmentation=get_segmentation,
    aug_args=aug_args,
    **train_args
)

shutil.copy(this_file, output_path)
