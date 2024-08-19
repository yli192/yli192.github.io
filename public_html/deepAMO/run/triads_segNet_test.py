import os
import sys
import numpy as np
import time
from keras.models import load_model, Model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append('/netscratch/garyli2/cnn_ho/')
from utils.segmentation_training import segmentation_testing
from utils.input_data import InputSegmentationArrays
from utils.kerasutils import get_image, correct_data_format, save_model_summary, get_channel_axis
from utils.image_reading import load_4Dimage_from_folder
"""Performs testing by a trained model."""

# When displaying figures have them as inline with notebook and not as standalone popups
#%matplotlib inline
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


input_path = '/netscratch/garyli2/cnn_ho_images/5yo_allSlices/'
image_directory = input_path + 'triads_images_to_train_segnet_largerDefectArea/'
output_path = ('/netscratch/garyli2/cnn_ho/experiments/outputs/'
              'triads_segmentation_netdepth3_try2_wWindowing_wMineAugOnlyforDef_woTruc_largerDefectArea_mixedloss/')
output_image_path = output_path + 'images/'

model_path = output_path + '/model/model.h5'

train_portion = 0.8
valid_portion = 0.2
image_array, label_array = load_4Dimage_from_folder(image_directory, (96,96), HE=False,Truc=False,Aug=False)
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

input_arrays = InputSegmentationArrays(image_train, label_train, image_valid, label_valid)
unique_labels = np.unique(label_valid)
print 'unique_labels: '
print unique_labels

# Workaround for custom loss function.
custom_objects = {
    'inner': lambda y_true, y_pre: y_pre,
#     'get_channel_axis': lambda: -1
}
net = load_model(model_path, custom_objects=custom_objects)


segmentation_model = Model(inputs=net.input, outputs=net.get_layer('segmentation').output)
#
# def get_segmentation(image):
#     return segmentation_model.predict(image).argmax(get_channel_axis())


def get_segmentation(image):
    return segmentation_model.predict(image)

segmentation_testing(
    get_segmentation=get_segmentation,
    input_arrays=input_arrays,
    output_path=output_path,
    output_image_path=output_image_path,
    rescale_types=None,
    rescale_probability=1.0,
    window_centers=None,
    window_widths=None,
    batch_size=1,
    save_image_batches=1,
    save_overlay=False,
    image_alpha=0.5,
    label_alpha=0.5,
    use_lumped_dice=True,
    unique_labels=unique_labels
)

