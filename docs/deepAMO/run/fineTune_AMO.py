import os,sys
sys.path.append('/netscratch/garyli2/cnn_ho/')
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from nets.custom_losses import GMM_loss
import tensorflow as tf
from utils.mdn import *
import shutil
from keras.utils import plot_model
from utils.classification_training_3slices_fineTuneOnly import classification_training
from utils.input_data import InputLists
from nets.simple_classification_net import fineTune_AMO_model7_AddtoGetFms_prjbcktocheckdefXYZ_usingdotproduct_6features_usingMDN_welu_wDO,fineTune_AMO_model6_AddtoGetFms_prjbcktocheckdefXYZ_usingdotproduct_6features_usingMDN_welu_wDO
if K.backend() == 'tensorflow':
    #print('111111')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #export CUDA_VISIBLE_DEVICES="1"
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())
    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1}, allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


if len(sys.argv) != 3 :
    print("Arguments: fineTune_AMO.py class_name batch_name")
    exit(1)

class_name = sys.argv[1]
batch_name = sys.argv[2]


input_path = '/netscratch/garyli2/cnn_ho_images/5yo_allSlices/'
output_path = ('/netscratch/garyli2/cnn_ho/experiments/outputs/'
              'fineTune_AMO_6features_5x2foldCrossValidation_singleModelForMultipleHOs/')
image_directory = output_path + '2ndStage_images_GL_wsoftmax_wrealImageNames_new/'

output_image_path = output_path + 'images/'

output_model_path = output_path + 'model2_wstd1p0_ft_to_GLJY_keepOrder_6features_doubleDU128dop5_'+str(class_name)+'_MDNw5Gaussians_wn1to1Label_welu_0Hidden_wRatingValueBtwn10to10_'+str(batch_name)+'/'
list_train = image_directory+ 'list_ft_train_5x2fcv_Gary_Junyu_combined_keepOrder_'+str(batch_name)+'_'+str(class_name)+'.txt'
list_valid = image_directory+ 'list_ft_valid_5x2fcv_Gary_Junyu_combined_keepOrder_'+str(batch_name)+'_'+str(class_name)+'.txt'
list_test = None

if output_model_path is not None and not os.path.exists(output_model_path):
    os.makedirs(output_model_path)

this_file = 'fineTune_AMO.py'
shutil.copy(this_file, output_model_path)
shutil.copy(list_train, output_path)
shutil.copy(list_valid, output_path)

input_lists = InputLists(
    directory=image_directory,
    list_train=list_train,
    list_valid=list_valid,
    list_test=list_test,
    class_mode='other'
)
numDenseUnits=128
numComponents=5
max_std=1.0
num_input_channels = 3

# load previous net and freeze the layers before the last layer
custom_objects = {
    'inner': lambda y_true, y_pre: y_pre,
    'tf': tf,
    'keras':keras,
    'elu_plus_one_plus_epsilon':elu_plus_one_plus_epsilon,
    'MixtureDensity':MixtureDensity,
    'get_mixture_loss_func':get_mixture_loss_func,
    "GMM_loss":GMM_loss,
    "mixture_loss":lambda y_true, y_pre: y_pre,
    "mdn_loss_func":lambda y_true, y_pre: y_pre
}

if os.path.exists(output_model_path+'model.h5'):
    print("previsou model exists, loading...")
    net = load_model(output_model_path+'model.h5', custom_objects=custom_objects)
    print("previsou model loaded")
else:
    net = fineTune_AMO_model7_AddtoGetFms_prjbcktocheckdefXYZ_usingdotproduct_6features_usingMDN_welu_wDO(numDenseUnits,0.5,numComponents,max_std)
    plot_model(net, show_shapes=True, show_layer_names=True, to_file=output_path + '/classification_net.pdf')

    # for layer in net.layers[:-5]:
    #     layer.trainable = False
    #
    # for layer in net.layers:
    #     print layer, layer.trainable

print("whole observer network:", net.summary())

#net.trainable = False
#net.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),loss=combine_loss([losses.mean_squared_error,losses.binary_crossentropy],[0.5,0.5]),metrics=['mae'])
#net.compile(optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),loss=GMM_loss(numComponents),metrics=['mae'])
net.compile(optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),loss=get_mixture_loss_func(1, numComponents),metrics=['mae'])



print(net.summary())

train_args = dict(
    output_path=output_path,
    input_lists=input_lists,
    output_model_path=output_model_path,
    output_image_path=output_image_path,
    rescale_labels=None,
    rescale_probability=1.0,
    window_centers=None,
    window_widths=None,
    use_pretrained_vgg=False,
    batch_size=72,
    n_batches_per_epoch=None,
    valid_batches_per_epoch=None,
    n_epochs=30000,
    print_freq=0.0,
    save_image_batches=1,
    class_weights_exp=0.5,
    #class_weights_normalize=False,
    #class_weights_scale=1.0,
    use_clahe=False,
    intensity_shift=0,
    model_choose_portion=0.1,
    restart_schedule=None,
    optimizer_str='Adam(lr=1e-3)',
    loss_str='get_mixture_loss_func',
    num_input_channels=num_input_channels
)

# Image intensity rescaling
use_windowing = False
use_clahe = False

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
        rotation_range=10,
        shift_range=[0.1, 0.1],
        zoom_range=[0.95, 1.05],
        channel_shift_range=0.,
        fill_mode='constant',
        cval=0.,
        flip=None,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        augmentation_probability=0.8,
        ndim=4  # Input array size. 4 for 3channel image
    )
    print(aug_args)
else:
    aug_args = None

rescale_types = None
# Save the best model based on validation accuracys
classification_training(
    net=net,
    rescale_types=rescale_types,
    aug_args=aug_args,
    **train_args
)