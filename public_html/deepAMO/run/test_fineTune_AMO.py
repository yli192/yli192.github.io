import os
import sys
import numpy as np
import time
from keras.models import load_model, Model
from keras import backend as K
import tensorflow as tf
from nets.custom_losses import GMM_loss
sys.path.append('/netscratch/garyli2/cnn_ho/')
from utils.input_data import InputLists
from utils.classification_testing_nhot import get_blocks

"""Performs testing by a trained model."""

# When displaying figures have them as inline with notebook and not as standalone popups
#%matplotlib inline
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #K.set_session(sess)


input_path = '/netscratch/garyli2/cnn_ho_images/5yo_allSlices/'
output_path = ('/netscratch/garyli2/cnn_ho/experiments/outputs/'
              'fineTune_AMO_6features_5foldCrossValidation_singleModelForMultipleHOs/')
image_directory = output_path + '2ndStage_images_GL_wsoftmax_wrealImageNames_new/'

output_image_path = output_path + 'images/'

input_model_path = output_path + 'model_ft_to_GLJY_keepOrder_6featuresTo1feature_doubleDU512dop7_class0_MDNw7Gaussians_wn1to1Label_welu_0Hidden_wRatingValueBtwn10to10_batch0/'
list_train = image_directory+ 'list_ft_train_5fcv_Gary_Junyu_combined_keepOrder.batch0_class0.txt'
list_valid = image_directory+ 'list_ft_valid_5fcv_Gary_Junyu_combined_keepOrder.batch0_class0.txt'
list_test = None

list_train_original = image_directory+ 'list_ft_train_5fcv_Gary_Junyu_combined_keepOrder.batch0_class0.txt'
list_valid_original = image_directory+ 'list_ft_valid_5fcv_Gary_Junyu_combined_keepOrder.batch0_class0.txt'

model_path = input_model_path + '/model.h5'
test_path = output_path + '/test_ft_to_GLJY_keepOrder_6featuresTo1feature_doubleDU512dop7_class0_MDNw7Gaussians_wn1to1Label_welu_0Hidden_wRatingValueBtwn10to10_batch0'

if test_path is not None and not os.path.exists(test_path):
    os.makedirs(test_path)

batch_size = 1
block_shape = None
target_image_size = (96, 96)
num_input_channels = 3
class_mode = 'other'
numComponents =7
from nets.custom_losses import categorical_focal_loss,MDN_loss, mean_loss,std_loss
from utils.mdn import *

def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_pi, out_sigma,out_mu  = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss
    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func

# Workaround for custom loss function.


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return keras.backend.elu(x) + 1 + keras.backend.epsilon()

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

#from nets.simple_classification_net import fineTune_AMO_model6_AddtoGetFms_prjbcktocheckdefXYZ_usingdotproduct_6features_usingMDN

#model = fineTune_AMO_model6_AddtoGetFms_prjbcktocheckdefXYZ_usingdotproduct_6features_usingMDN(64,0.7,10)
model = load_model(model_path,custom_objects)
print(model.summary())
print('Model loaded.')

feature_vector = model.get_layer('concatenate_1').output
feature_vector_model = Model(inputs=model.inputs,outputs=feature_vector)


inputs = InputLists(directory=image_directory, list_train=list_train, list_valid=list_valid, list_test=list_test,
                    class_mode=class_mode)
inputs_original = InputLists(directory=image_directory, list_train=list_train_original, list_valid=list_valid_original, list_test=list_test,
                    class_mode=class_mode)
if not os.path.exists(test_path):
    os.makedirs(test_path)

start_time = time.time()

# Perform testing
label_true = []
label_all = []
pred_all = []
test_num_batches = inputs.get_test_num_batches(batch_size)
n_batches = 0

for im_name in inputs_original.valid:
    ur_num = int(im_name.strip().split('.')[0][2:])
    if ur_num < 192:
        label_gt = 0
    else:
        label_gt = 1
    #print im_name, label_gt
    label_true.append(int(label_gt))


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def generate(output, testSize, numComponents=5, outputDim=1, M=1):
    out_pi = output[:,:numComponents]
    out_sigma = output[:,numComponents:2*numComponents]
    out_mu = output[:,2*numComponents:]
    out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
    out_mu = np.transpose(out_mu, [1,0,2])
# use softmax to normalize pi into prob distribution
#     max_pi = np.amax(out_pi, 1, keepdims=True)
#     out_pi = out_pi - max_pi
#     out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = np.exp(out_sigma)
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)
    #print("pis:",out_pi)
    mu = 0
    std = 0
    idx = 0
    top_3_pi_idx = out_pi.argsort()[0][-3:][::-1]
    top_3_pi=np.take(out_pi,top_3_pi_idx)
    top_3_mu=np.take(out_mu,top_3_pi_idx)
    top_3_sigma = np.take(out_sigma, top_3_pi_idx)
    print("top_3_pi_idx",top_3_pi_idx)
    print("top_3_pi",top_3_pi)
# use softmax to normalize pi into prob distribution
    max_pi = np.amax(top_3_pi, 0, keepdims=True)
    top_3_pi = top_3_pi - max_pi
    top_3_pi = np.exp(top_3_pi)
    normalize_pi = 1 / (np.sum(top_3_pi, 0, keepdims=True))
    top_3_pi = normalize_pi * top_3_pi
    print("top_3_pi_after_normalization", top_3_pi)

    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                idx = np.random.choice(3, 1, p=top_3_pi)
                print("mixing coefficients:",out_pi)
                print("mus:", out_mu)
                print('sigmas:',out_sigma)
                print('sampled index', idx)
                #idx = np.argmax(out_pi[i])
                #print(idx)
                mu = top_3_mu[idx]
                #std = out_sigma[i, idx]
                #std = top_3_sigma[idx]
                #result[i, j, d] = mu + rn[i, j]*std
                #print('final mu', mu)
                result[i, j, d] = mu

    return result

for image, label in inputs.get_valid_flow_3slices(batch_size, False, target_image_size, num_input_channels=num_input_channels):
    if block_shape is not None:
        image = get_blocks(image, block_shape=block_shape)

    pred = generate(model.predict(image), batch_size, numComponents)
    fv=feature_vector_model.predict(image)
    print('feature vector:',fv)
    #pred = model.predict(image, batch_size=batch_size)
    print("final pred",pred)
    print('human rating',label)
    label_all.append(label)
    #print pred[0][0]
    pred_all.append(pred[0][0][0])
    n_batches += 1
    sys.stdout.write('Testing n_batches: %d\r' % n_batches)
    sys.stdout.flush()
    if n_batches == test_num_batches:
        break

label_all = np.concatenate(label_all)
#pred_all = np.concatenate(pred_all)


total_difference = 0
for i in range(len(label_all)):
    abs_difference = np.abs(label_all[i] - pred_all[i])
    print(label_all[i],pred_all[i], abs_difference)
    total_difference = abs_difference + total_difference
    mean_abs_difference = total_difference/len(label_all)



import sklearn
label_true = np.asarray(label_true)
print(label_true)
print(len(label_true))
print(label_all)
print(pred_all)
print("mean absolute difference:\n", mean_abs_difference)
print("mutual information:\n", sklearn.metrics.mutual_info_score(label_all, pred_all))
print(sklearn.metrics.mutual_info_score([1], [1]))
#print inputs.valid

np.savez_compressed(os.path.join(test_path, 'labels_and_preds_ho.npz'), label_all=label_true, pred_all=label_all)
np.savez_compressed(os.path.join(test_path, 'labels_and_preds_cnn.npz'), label_all=label_true, pred_all=pred_all)

end_time = time.time()

print('\nTime used: ', end_time - start_time)
print('Done.')