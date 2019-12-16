import os, sys, random
import numpy as np
import scipy as sp
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn

import rpy2
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.conversion import rpy2py as r2py

from sklearn.manifold import TSNE
import umap

# makes heatmap for 6 cluster data
def make_heatmap6(sim_true_norm, sim_raw_norm, dca_zinb_norm):
    # we didn't filter zero-genes earlier, to allow model to work with data w diff dropout %s
    sc.pp.filter_genes(sim_raw_norm,min_counts=1)
    sim_true_norm = sim_true_norm[:, sim_raw_norm.var_names]
    dca_zinb_norm = dca_zinb_norm[:, sim_raw_norm.var_names]

    de_genes = np.where(sim_true_norm.var.loc[:, 'DEFacGroup1':'DEFacGroup6'].values.sum(1) != 6.0)[0]

    obs_idx = np.random.choice(list(range(sim_raw_norm.n_obs)), 300, replace=False)
    idx = np.argsort(sim_true_norm.obs.Group.values[obs_idx])
    obs_idx = obs_idx[idx]

    plt.figure(1)

    plt.clf()
    grd = seaborn.clustermap(sim_true_norm.X[:, de_genes][obs_idx, :],standard_scale=1,cmap="coolwarm")
    plt.savefig('figures/heatmap_6group_left.png')

    plt.clf()
    grd = seaborn.clustermap(sim_raw_norm.X[:, de_genes][obs_idx, :],standard_scale=1,cmap="coolwarm")
    plt.savefig('figures/heatmap_6group_middle.png')

    plt.clf()
    grd = seaborn.clustermap(dca_zinb_norm.X[:, de_genes][obs_idx, :],standard_scale=1,cmap="coolwarm")
    plt.savefig('figures/heatmap_6group_right.png')


def make_heatmap2(sim_true_norm, sim_raw_norm, dca_zinb_norm):
    # get indices of differentially expressed genes (given by simulator)
    de_genes = np.where(sim_true_norm.var.loc[:, 'DEFacGroup1':'DEFacGroup2'].values.sum(1) != 2.0)[0]

    # get indices of 300 random cells to display, then order by group
    obs_idx = np.random.choice(list(range(sim_raw_norm.n_obs)), 300, replace=False) #changed from sim_raw_norm
    idx = np.argsort(sim_true_norm.obs.Group.values[obs_idx])
    obs_idx = obs_idx[idx]

    plt.figure(1)

    plt.clf()
    grd = seaborn.clustermap(sim_true_norm.X[:, de_genes][obs_idx, :],standard_scale=1,cmap="coolwarm")
    plt.savefig('figures/heatmap_2group_left.png')

    plt.clf()
    grd = seaborn.clustermap(sim_raw_norm.X[:, de_genes][obs_idx, :],standard_scale=1,cmap="coolwarm")
    plt.savefig('figures/heatmap_2group_middle.png')

    plt.clf()
    grd = seaborn.clustermap(dca_zinb_norm.X[:, de_genes][obs_idx, :],standard_scale=1,cmap="coolwarm")
    plt.savefig('figures/heatmap_2group_right.png')

# Uses r2py to connect to Splatter R package and simulate data
# Returns a list of two AnnData objects: second is just first before dropouts
# numgroups can be either 2 or 6
def get_data(numgroups):
    with localconverter(ro.default_converter + pandas2ri.converter):
        if numgroups == 2:
            r.source('~/Documents/rscripts/splatter-2.R')
        elif numgroups == 6:
            r.source('~/Documents/rscripts/splatter-6.R')
        counts = r2py(r['counts']) # cell-by-gene dataframe
        cellinfo = r2py(r['cellinfo']) # Cell, Batch, Group
        geneinfo = r2py(r['geneinfo']) # Gene

        sim = sc.AnnData(counts.values, obs=cellinfo, var=geneinfo)
        sim.obs_names = cellinfo.Cell
        sim.var_names = geneinfo.Gene
        if numgroups == 2:
            sc.pp.filter_genes(sim,min_counts=1) # omitted in 6 case so we can generalize to diff dropout %s

        truecounts = r2py(r['truecounts'])
        dropout = r2py(r['dropout'])
        print("percent dropout: {}".format(np.sum(dropout.values)/(sim.n_obs*sim.n_vars)))

        sim_true = sc.AnnData(truecounts.values, obs=cellinfo, var=geneinfo)
        sim_true.obs_names = cellinfo.Cell
        sim_true.var_names = geneinfo.Gene
        sim_true = sim_true[:, sim.var_names]

        return [sim, sim_true]

def vis_data(sim_true, sim, adata, numgroups):
    def embed_pca(A_):
        A = A_.copy()
        sc.pp.normalize_total(A)
        sc.pp.log1p(A)
        sc.pp.pca(A)
        return A

    sim = embed_pca(sim)
    sim_true = embed_pca(sim_true)
    adata = embed_pca(adata)

    ################################################################################
    # # USED FOR CLUSTERING / ARI
    #
    # data = sim
    # print("running neighbors")
    # sc.pp.neighbors(data)
    # print("running clustering")
    # sc.tl.leiden(data)
    #
    # print("LOUVAINPRINT1")
    # print(data.obs['leiden'])
    # print(type(data.obs['leiden']))
    # print("LOUVAINPRINT2")
    # print(data.obs['leiden'].values)
    # print(type(data.obs['leiden'].values))
    #
    # print("GROUPPRINT1")
    # print(data.obs['Group'])
    # print(type(data.obs['Group']))
    # print("GROUPPRINT2")
    # print(data.obs['Group'].values)
    # print(type(data.obs['Group'].values))
    # return
    #
    # import sklearn.metrics.adjusted_rand_score as ARI
    # print("running ARI")
    # print(ARI(labels_true=data.obs['Group'].values, labels_pred=data.obs['leiden'].values))
    #
    # return
    ########################################################################################


    # # make heatmaps
    # if (numgroups == 2):
    #     make_heatmap2(sim_true,sim,adata)
    # if (numgroups == 6):
    #     make_heatmap6(sim_true,sim,adata)


    # C = [[random.random() for i in range(3)] for i in range(numgroups)]
    if (numgroups == 2):
        C = [[0.15, 0.5, 0.67], [0.76, 0.38, 0.32]] # just some nice colors
    else:
        C = [[0.11, 0.23, 0.94], [0.65, 0.52, 0.78], [0.49, 0.7, 0.05], [0.87, 0.17, 0.38], [0.49, 0.96, 0.62], [0.27, 0.27, 0.14]]

    if (numgroups > 1):
        group_names = np.unique(sim.obs['Group'])
        assert(len(group_names) == numgroups)
        colorvec = [C[np.where(group_names == group_name)[0][0]] for group_name in sim.obs['Group']]
    else:
        colorvec = [[0.15, 0.5, 0.67] for s in range(len(sim.X))]


    def plot_(type_):
        if numgroups < 3:
            # print components from PCA
            plt.scatter(type_.obsm['X_pca'][:,0], type_.obsm['X_pca'][:,1], c=colorvec, s=3)
        else:
            # print components from UMAP
            print("running UMAP")
            embedding = umap.UMAP(n_neighbors=5,
                          min_dist=0.001,
                          metric='euclidean').fit_transform(type_.obsm['X_pca'])
            plt.scatter(embedding[:,0],embedding[:,1],c=colorvec,s=3)


    plt.clf()
    plot_(sim_true)
    plt.savefig('figures/clusters_' + str(numgroups) + 'group_left.png')

    plt.clf()
    plot_(sim)
    plt.savefig('figures/clusters_' + str(numgroups) + 'group_middle.png')

    plt.clf()
    plot_(adata)
    plt.savefig('figures/clusters_' + str(numgroups) + 'group_right.png')
    

    # type = sim_true_norm
    # plt.scatter(type.obsm['X_pca'][:,0], type.obsm['X_pca'][:,1], c=colorvec, s=1)
    #
    # plt.subplot(122)
    # type = sim_norm
    # plt.scatter(type.obsm['X_pca'][:,0], type.obsm['X_pca'][:,1], c=colorvec, s=1)
    # plt.show()

    # ##################
    # # if TSNE desired instead of UMAP
    # # tsne = TSNE(perplexity=30,n_iter=1000)
    # #
    # # embedding = tsne.fit_transform(sim.obsm['X_pca'])
    # # plt.scatter(embedding[:,0],embedding[:,1],s=1)
    # # plt.show()
    # # print(tsne.kl_divergence_)
    # # print(tsne.n_iter_)
    # #
