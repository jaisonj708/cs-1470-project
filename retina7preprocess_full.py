from matplotlib import pyplot as plt
from matplotlib import colors
import sklearn.decomposition
import tensorflow as tf
import numpy as np
import math
import umap
from sklearn.manifold import TSNE


with open("/Users/jaisonjain/Downloads/GSM1626799_P14Retina7.txt") as f1, \
    open("/Users/jaisonjain/Downloads/GSM1626798_P14Retina6.txt") as f2, \
    open("/Users/jaisonjain/Downloads/GSM1626797_P14Retina5.txt") as f3, \
    open("/Users/jaisonjain/Downloads/GSM1626796_P14Retina4.txt") as f4, \
    open("/Users/jaisonjain/Downloads/P14Retina_merged.txt") as f:

    num_test_genes = np.inf

    # def get_genes(file):
    #     file.readline()
    #     line = file.readline()
    #     vals = []
    #     while (line != '' and line != '\n'):
    #         vals += [line.split()[0]]
    #         line = file.readline()
    #     return set(vals)

    print("loading data...")
    f.readline()
    gene_levels = []
    i = 0
    line = f.readline()
    while (line != '' and line != '\n' and i < num_test_genes):
        vals = [int(str) for str in line.split()[1:]]
        gene_levels.append(vals)
        line = f.readline()
        i += 1

    gene_levels = np.array(gene_levels)
    print(gene_levels.shape)

    # normalize by total UMI
    print('normalizing by total transcripts...')
    cells = np.transpose(gene_levels)
    cells = np.stack([c/(np.sum(c)+10e-10) for c in cells]) * 10e4

    # # print num UMIs by barcode
    kvs = dict(enumerate(np.sum(cells,axis=-1))).items()
    sorted_ = sorted(kvs, key= lambda kv: (kv[1], kv[0]), reverse=True)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(len(sorted_)), [kv[1] for kv in sorted_], marker='.')
    plt.show()

    # take only high-expressing cells
    print('filtering out low-expressing cells...')
    min_genes = round(cells.shape[1]/20)
    if num_test_genes == np.inf: min_genes = 900
    cells = np.stack([c for c in cells if np.sum([int(g > 0) for g in c]) >= min_genes])
    print('number of cells: {}'.format(len(cells)))
    # cells = np.stack([[np.log(g+10e-10) for g in c] for c in cells])

    # take log here?
    # take only high-variance genes (bucketing by expression range here, not num genes)
    print('finding high-variance genes... (sorting mean expr vals)')
    num_bins = 20 # or round(len(gene_levels)/1000)
    mean_expr = np.mean(cells,axis=0)
    min = np.min(mean_expr)
    max = np.max(mean_expr)
    step = (max - min) / num_bins

    kvs = dict(enumerate(mean_expr)).items()
    sorted_ = sorted(kvs, key= lambda kv: (kv[1], kv[0]))

    # find high variance genes
    print('finding high-variance genes... (bucketing and selecting mean expr vals)')
    high_variance_genes = []
    vals = []
    curr_min = min
    for kv in sorted_:
        if kv[1] <= curr_min + step:
            vals.append(kv)
        else:
            mean = np.mean([kv[1] for kv in vals])
            std = np.std([kv[1] for kv in vals]) + 10e-10
            high_variance_genes += [kv[0] for kv in vals if (kv[1] - mean)/std > 1.7]

            vals = [kv]
            curr_min = min + step * round((kv[1] - min) / step)
    high_variance_genes = sorted(high_variance_genes)

    # take only high-variance genes
    print('filtering out low-variance genes...')
    cells_gene_filtered = np.take(cells,high_variance_genes,axis=1)
    print(cells_gene_filtered.shape)
    cells = cells_gene_filtered


    # # pca (no standardization, just taking log)
    # print('running pca... (taking log)')
    # cells = np.stack([[np.log(g+10e-10) for g in c] for c in cells]) # take log
    #
    # print('running pca... (eigen)')
    # S = np.cov(np.transpose(cells))
    # eigvals,eigvecs = np.linalg.eig(S)
    # indices = np.flip(np.argsort(eigvals))
    # eigvals = np.take(eigvals, indices)
    # eigvecs = np.take(eigvecs, indices, axis=1)
    #
    # print('running pca... (projecting)')
    # def projection_magn(v,w):
    #     if (np.dot(w,w) <= 0): print("bad")
    #     len_w = math.sqrt(np.dot(w,w))
    #     return np.dot(v,w)/len_w
    # x = [projection_magn(c, eigvecs[:,0]) for c in cells]
    # y = [projection_magn(c, eigvecs[:,1]) for c in cells]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(x,y, marker='.')
    # plt.show()
    #
    # plt.plot(np.arange(35), eigvals[:35], marker='.')
    # plt.show()

    # pca
    print("running pca... (standardizing)")
    means = np.mean(cells,axis=0)
    stds = np.std(cells,axis=0)
    cells = np.divide(cells-means, stds)

    print("running pca... (getting eigens)")
    ret = sklearn.decomposition.PCA(n_components=32).fit(cells)
    print(ret.explained_variance_ratio_)

    print("running pca... (transforming)")
    cells = sklearn.decomposition.PCA(n_components=32).fit_transform(cells)
    print(cells.shape)

    # # UMAP
    print("running umap...")
    embedding = umap.UMAP(n_neighbors=10,
                          min_dist=0.001,
                          metric='euclidean').fit_transform(cells)
    plt.scatter(embedding[:,0],embedding[:,1],s=1)
    plt.show()

    # TSNE
    print("running tsne...")
    embedding = TSNE(perplexity=30,n_iter=2000).fit_transform(cells)
    plt.scatter(embedding[:,0],embedding[:,1],s=1)
    plt.show()
