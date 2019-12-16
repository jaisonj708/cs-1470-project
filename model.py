import os, argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Lambda
from tensorflow.keras import Model
import numpy as np
import scanpy as sc


class ZINBAutoencoderWrapper(Model):
    def __init__(self, num_genes):
        super(ZINBAutoencoderWrapper, self).__init__()

        self.batch_size = 32
        self.hidden_sizes = [64,32,64]
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

        self.model = None
        self.extra_models = {}
        self.inputs = None
        self.pi = None
        self.theta = None
        self.mu = None
        self.latent = None

        input = Input(shape=(num_genes,))
        size_factors = Input(shape=(1,))
        self.inputs = [input, size_factors]

        out = input
        for i,l in enumerate(self.hidden_sizes):
            out = Dense(l,name='my-dense%s' % i)(out)
            out = BatchNormalization(center=True, scale=False,name='my-batchnorm%s' % i)(out)
            out = Activation('relu',name='my-relu%s' % i)(out)
            if (i == math.floor(len(self.hidden_sizes)/2)):
                self.latent = out

        self.pi = Dense(num_genes, activation='sigmoid',name='pi-layer')(out)

        theta = Dense(num_genes, name='theta-layer')(out)
        self.theta = Lambda(lambda x: tf.clip_by_value(tf.math.softplus(x), 1e-4, 1e4), name='theta-layer2')(theta)

        mu = Dense(num_genes,name='mu-layer')(out)
        mu = Lambda(lambda x: tf.clip_by_value(tf.math.exp(x), 1e-5, 1e6, name='mu-layer2'))(mu)
        self.mu = Lambda(lambda x: tf.multiply(x[0], tf.reshape(x[1],(-1,1))), name='mu-layer3')([mu, size_factors])

        self.model = Model(inputs=self.inputs, outputs=[self.pi,self.theta,self.mu])
        self.extra_models['pi'] = Model(inputs=input, outputs=self.pi)
        self.extra_models['theta'] = Model(inputs=input, outputs=self.theta)
        self.extra_models['encoder'] = Model(inputs=input, outputs=self.latent)

        self.model.compile(optimizer=self.optimizer,loss=tf.keras.losses.binary_crossentropy) # for graph building purposes

    # Returns model output (list of 3 tensors, each with dimensions batch_size x num_genes)
    # given input: list of 2 tensors/arrays with dims (batch_size x num_genes) and (batch_size,)
    def call(self, inputs):
        return self.model.predict_on_batch(inputs)

    # Returns loss: negative log ZINB-likelihood of y_true given: y_pred (mu), pi, theta
    # All four of these are tensors/arrays of dims (batch_size x num_genes)
    def loss_function(self, y_pred, y_true, pi, theta):
        eps = 1e-10
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        theta = tf.minimum(theta, 1e6)

        nb_case = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
        nb_case += (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps)))
        nb_case += (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))
        nb_case -= tf.math.log(1.0-pi+eps)

        zero_nb = tf.math.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -tf.math.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = tf.where(tf.math.less(y_true, 1e-8), zero_case, nb_case)

        return result

    # From adata, predict theta and pi params, as well as mu/mean* and latent* vector
    # Insert prediction into adata object as annotations
    def predict(self, adata):
        adata.obsm['X_theta'] = self.extra_models['theta'].predict_on_batch(adata.X)
        adata.obsm['X_pi'] = self.extra_models['pi'].predict_on_batch(adata.X)

        _,_,mu = self.model.predict_on_batch([adata.X, adata.obs.size_factors.to_numpy()])
        adata.X = mu.numpy()

        adata.obsm['X_latent'] = self.extra_models['encoder'].predict_on_batch(adata.X)

        return adata

class AutoencoderWrapper(Model):
    def __init__(self, num_genes):
        super(AutoencoderWrapper, self).__init__()

        self.batch_size = 32
        self.hidden_sizes = [64,32,64]
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self.extra_models = {}

        input = Input(shape=(num_genes,))
        size_factors = Input(shape=(1,))
        self.inputs = [input, size_factors]

        out = input
        for i,l in enumerate(self.hidden_sizes):
            out = Dense(l,name='my-dense%s' % i)(out)
            out = BatchNormalization(center=True, scale=False,name='my-batchnorm%s' % i)(out)
            out = Activation('relu',name='my-relu%s' % i)(out)
            if (i == math.floor(len(self.hidden_sizes)/2)):
                self.latent = out

        mu = Dense(num_genes,name='mu-layer')(out)
        mu = Lambda(lambda x: tf.clip_by_value(tf.math.exp(x), 1e-5, 1e6, name='mu-layer2'))(mu)
        self.mu = Lambda(lambda x: tf.multiply(x[0], tf.reshape(x[1],(-1,1))), name='mu-layer3')([mu, size_factors])
        self.model = Model(inputs=self.inputs, outputs=[self.mu,self.mu,self.mu])

        self.extra_models['encoder'] = Model(inputs=input, outputs=self.latent)

        self.model.compile(optimizer=self.optimizer,loss=tf.keras.losses.binary_crossentropy) # for graph building purposes

    def call(self, inputs):
        return self.model.predict_on_batch(inputs)

    def loss_function(self, y_pred, y_true, pi, theta):
        return tf.reduce_mean(tf.square(y_true-y_pred))

    def predict(self, adata):
        _,_,mu = self.model.predict_on_batch([adata.X, adata.obs.size_factors.to_numpy()])
        adata.X = mu.numpy()

        adata.obsm['X_latent'] = self.extra_models['encoder'].predict_on_batch(adata.X)

        return adata
