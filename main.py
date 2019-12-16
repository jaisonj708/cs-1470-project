import os, argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Lambda
from tensorflow.keras import Model
import numpy as np
import scanpy as sc
import pandas as pd
from model import ZINBAutoencoderWrapper, AutoencoderWrapper
from anndata import AnnData
import splatter
import matplotlib.pyplot as plt
import matplotlib
import time

# Returns an ArgumentParser object with input file (.txt file, format:
# non-transformed gene-by-cell matrix with labels), output file, splatter bool
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='.txt file with '
    'non-transformed gene-by-cell matrix (with labels)')
    parser.add_argument('--output_file', type=str, default='output.txt', help =
    '.txt file for output')
    parser.add_argument('--splatter', dest='splatter', action='store_true', help =
    'splatter file?')

    parser.set_defaults(splatter=False)

    return parser.parse_args()

# Trains the model (in wrapper) using normalized cell-by-gene matrix (inputs),
# corresponding size factors (size_factors), and unnormalized matrix (output)
def train(wrapper, inputs, size_factors, labels):

    index = 0
    while (index + wrapper.batch_size <= labels.shape[0]):
        inputs_ = [inputs[index:index + wrapper.batch_size], size_factors[index:index + wrapper.batch_size]]
        labels_ = labels[index:index + wrapper.batch_size]

        with tf.GradientTape() as tape:
            pi,theta,mu = wrapper(inputs_)
            loss = wrapper.loss_function(mu, labels_, pi, theta)

            if (index % (wrapper.batch_size * 15) == 0):
                print(tf.reduce_mean(loss))

        grads = tape.gradient(loss, wrapper.model.trainable_variables)
        wrapper.model.optimizer.apply_gradients(zip(grads, wrapper.model.trainable_variables))

        index += wrapper.batch_size

def mse(A,B):
    return(np.mean(np.square(A - B)))

def main():

    NUM_GROUPS = 2 # relevant for Splatter-generated data, set to 1 if from text file
    TRAINP = True
    FIRST = False
    num_epochs = 1

    args = get_parser()
    if args.splatter:
        adata, adata_true = splatter.get_data(NUM_GROUPS)
        sc.pp.filter_cells(adata, min_counts=1)
        adata_true = adata_true[adata.obs_names, :]

        sim = adata.copy() # data before inference
        print(mse(adata.X,adata_true.X))
    else:
        # set first_column_names to True if row/column labels in the file
        adata = sc.read(args.input_file, first_column_names=False)
        adata = adata.transpose() # -> cell by gene

        # # collects data for just 200 genes from complete file -> separate file
        # with open('retina7_truncated.txt','w') as f:
        #     count = 0
        #
        #     for gene in np.transpose(adata.X):
        #         for val in gene:
        #             f.write(str(val) + "\t")
        #         f.write("\n")
        #
        #         max = 200
        #         if count > max:
        #             break;
        #         count += 1

        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        sim = adata.copy()
        adata_true = sim # adata_true dummy variable
        print(adata)
    print("Successfully read: {} cells and {} genes".format(adata.n_obs,adata.n_vars))

    # further preprocessing
    sc.pp.normalize_total(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    if NUM_GROUPS == 2:
        model_file = 'weights/2000.h5' # saved weights and biases, available upon request
    elif NUM_GROUPS == 6:
        model_file = 'weights/6000.h5'
    # model_file = "model_retina.h5"
    # model_file = 'saved_mods/model_mse_trained.h5'

    wrapper = ZINBAutoencoderWrapper(adata.n_vars)

    if FIRST:
        assert(os.path.getsize(model_file) == 0) # remove if you want to delete model each time (eg if testing hyperparameters)
        wrapper.model.save(model_file)
    del wrapper.model
    wrapper.model = tf.keras.models.load_model(model_file)

    t1 = time.time()
    if TRAINP:
        for n in range(num_epochs):
            train(wrapper, adata.X, adata.obs.size_factors.to_numpy(), sim.X)
            print(mse(wrapper.predict(adata.copy()).X, adata_true.X))
            wrapper.model.save(model_file)
        print(wrapper.model.summary())
    print("train time (sec) = {}".format(time.time()-t1))

    wrapper.predict(adata)

    if args.splatter:
        splatter.vis_data(adata_true, sim, adata, numgroups=NUM_GROUPS)
    else:
        splatter.vis_data(adata_true, sim, adata, numgroups=1)


if __name__ == '__main__':
   main()
