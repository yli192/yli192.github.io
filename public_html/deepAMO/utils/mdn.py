from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import math
from keras.layers import Dense, Input, merge,Concatenate
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def get_mixture_coef(output, numComonents=24, outputDim=1):
    out_pi = output[:,:numComonents]
    out_sigma = output[:,numComonents:2*numComonents]
    out_mu = output[:,2*numComonents:]
    out_mu = K.reshape(out_mu, [-1, numComonents, outputDim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu
    result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = K.prod(result, axis=[0])
    return result

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)

def mdn_loss(numComponents=24, outputDim=1):
    def loss(y, output):
        #print y
        out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss
def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return keras.backend.elu(x) + 1 + keras.backend.epsilon()

class MixtureDensity(Layer):
    #print Layer
    def __init__(self, kernelDim, numComponents,hiddenDim, **kwargs):
        self.hiddenDim = hiddenDim
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        # with tf.name_scope('MDN'):
        #     self.mdn_mus = Dense(self.numComponents * self.kernelDim, name='mdn_mus')  # mix*output vals, no activation
        #     self.mdn_sigmas = Dense(self.numComponents * self.kernelDim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
        #     self.mdn_pi = Dense(self.numComponents, name='mdn_pi')  # mix vals, logits
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inputDim = input_shape[1]
        self.outputDim = self.numComponents * (2+self.kernelDim)
        self.Wh = K.variable(np.random.normal(scale=0.5,size=(self.inputDim, self.hiddenDim)))
        self.bh = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim)))
        self.Wo = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5,size=(self.outputDim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]
        #print input_shape
    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output
        # with tf.name_scope('MDN'):
        #     mdn_out = Concatenate([self.mdn_mus(x),
        #                                   self.mdn_sigmas(x),
        #                                   self.mdn_pi(x)],
        #                                  name='mdn_outputs')
        # return mdn_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.outputDim)

    def get_config(self):
        config = {
            "kernelDim": self.kernelDim,
            "numComponents": self.numComponents,
            "hiddenDim":self.hiddenDim
        }
        base_config = super(MixtureDensity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


import keras
#from tensorflow import layers
from tensorflow.keras import layers
import numpy as np



class MDN(Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.
    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus')  # mix*output vals, no activation
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            self.mdn_pi = Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super(MDN, self).build(input_shape)

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = Concatenate([self.mdn_mus(x),
                                          self.mdn_sigmas(x),
                                          self.mdn_pi(x)],
                                         name='mdn_outputs')
        return mdn_out

    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_pi, out_sigma,out_mu  = tf.split(y_pred, num_or_size_splits=[num_mixes,
                                                                         num_mixes * output_dim,
                                                                         num_mixes * output_dim
                                                                         ],
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



class MDN_v2(Layer):
    def __init__(self, output_dim, num_mix, kernel='unigaussian', **kwargs):
        self.output_dim = output_dim
        self.kernel = kernel
        self.num_mix = num_mix

        with tf.name_scope('MDNLayer'):
            # self.inputs      = Input(shape=(input_dim,), dtype='float32', name='msn_input')
            self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus')  # (self.inputs)
            self.mdn_sigmas = Dense(self.num_mix, activation=K.exp, name='mdn_sigmas')  # (self.inputs)
            self.mdn_pi = Dense(self.num_mix, activation=K.softmax, name='mdn_pi')  # (self.inputs)
            # self.mdn_out     = merge([self.mdn_mus, self.mdn_sigmas, self.mdn_pi], mode='concat', name='mdn_out')

        super(MDN_v2, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.input_shape = input_shape

        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)

        self.trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

        self.non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        # self.updates = self.mdn_mus.updates + self.mdn_sigmas.updates + self.mdn_pi.updates
        # self.regularizers = self.mdn_mus.regularizers + self.mdn_sigmas.regularizers + self.mdn_pi.regularizers
        # self.constraints = self.mdn_mus.constraints + self.mdn_sigmas.constraints + self.mdn_pi.constraints

        self.built = True

    def call(self, x, mask=None):
        m = self.mdn_mus(x)
        s = self.mdn_sigmas(x)
        p = self.mdn_pi(x)

        with tf.name_scope('MDNLayer'):
            mdn_out = Concatenate([m, s, p], name='mdn_out')
        return mdn_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  #   'input_shape': self.input_shape,
                  'num_mix': self.num_mix,
                  'kernel': self.kernel}
        base_config = super(MDN_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_loss_func(self):
        def unigaussian_loss(y_true, y_pred):
            mix = tf.range(start=0, limit=self.num_mix)
            out_mu, out_sigma, out_pi = tf.split_v(split_dim=1,
                                                   size_splits=[self.num_mix * self.output_dim, self.num_mix,
                                                                self.num_mix], value=y_pred, name='mdn_coef_split')

            # tf.to_float(out_mu)
            # print('----- ', tf.shape(y_pred)[0].eval(session=K.get_session()))
            # print('----- ', tf.shape(y_pred)[1])/

            def loss_i(i):
                batch_size = tf.shape(out_sigma)[0]
                sigma_i = tf.slice(out_sigma, [0, i], [batch_size, 1], name='mdn_sigma_slice')
                pi_i = tf.slice(out_pi, [0, i], [batch_size, 1], name='mdn_pi_slice')
                mu_i = tf.slice(out_mu, [0, i * self.output_dim], [batch_size, self.output_dim], name='mdn_mu_slice')

                print('***.....>> ', i * self.output_dim)
                tf.Print(mu_i, [i], ">>>>>>>  ")
                # print('.....>> ', tf.shape(y_true))

                dist = tf.contrib.distributions.Normal(mu=mu_i, sigma=sigma_i)
                loss = dist.pdf(y_true)

                # loss = gaussian_kernel_(y_true, mu_i, sigma_i)

                loss = pi_i * loss

                return loss

            result = tf.map_fn(lambda m: loss_i(m), mix, dtype=tf.float32, name='mix_map_fn')

            result = tf.reduce_sum(result, axis=0, keepdims=False)
            result = -tf.log(result)
            # result = tf.reduce_mean(result, axis=1)
            result = tf.reduce_mean(result)
            # result = tf.reduce_sum(result)
            return result

        if self.kernel == 'unigaussian':
            with tf.name_scope('MDNLayer'):
                return unigaussian_loss