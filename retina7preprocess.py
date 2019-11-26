import numpy as np
import math

with open("/Users/jaisonjain/Downloads/GSM1626799_P14Retina7.txt") as f:
    num_test_genes = 1000

    f.readline()
    gene_levels = []
    i = 0
    line = f.readline()
    while (line != '' and line != '\n' and i <= num_test_genes):
        vals = [int(str) for str in line.split()[1:]]
        gene_levels.append(vals)
        line = f.readline()
        i += 1

    cells = np.transpose(gene_levels)

    cells = np.stack([c/(np.sum(c)+10e-10) for c in cells]) * 10e5 # normalize cells

    min_genes = round(len(gene_levels)/20)
    cells = np.stack([c for c in cells if np.sum([int(g > 0) for g in c]) >= min_genes]) # filter out low-expressing cells

    # take log here?
    # take only high-variance genes (bucketing by expression range here, not num genes)
    num_bins = 20 # round(len(gene_levels)/1000)
    mean_expr = np.mean(cells,axis=0)
    min = np.min(mean_expr)
    max = np.max(mean_expr)
    step = (max - min) / num_bins

    kvs = dict(enumerate(mean_expr)).items()
    sorted = sorted(kvs, key= lambda kv: (kv[1], kv[0]))

    high_variance_genes = []
    vals = []
    curr_min = min
    for kv in sorted:
        if kv[1] <= curr_min + step:
            vals.append(kv)
        else:
            mean = np.mean([kv[1] for kv in vals])
            std = np.std([kv[1] for kv in vals]) + 10e-10
            high_variance_genes += [kv[0] for kv in vals if (kv[1] - mean)/std > 1.7]

            vals = [kv]
            curr_min = min + step * round((kv[1] - min) / step)

    print(high_variance_genes)


    # filter out non-high-variance data, normalize features (mean 0, std constant), run umap/NN/pca etc
