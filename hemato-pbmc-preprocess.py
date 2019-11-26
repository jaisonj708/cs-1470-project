#!/usr/bin/python

import numpy as np
import pandas as pd

##### CORTEX dataset #####
cortex = np.array(pd.read_csv('expression_mRNA_17-Aug-2014.txt', sep='\t', header=None, skiprows=11))
cortex_all = cortex[:,2:]
gene_vars = cortex_all.var(axis=1)
gene_vars_ordered_idx = np.argsort(gene_vars)[::-1]
genes_most_varied = gene_vars_ordered_idx[:558]
cortex_most_varied = cortex_all[cortex]
np.savetxt('cortex_mostvar_unnorm.txt',cortex_most_varied, fmt='%i')

##### HEMATO dataset #####
hemato = np.array(pd.read_csv('GSM2388072_basal_bone_marrow.filtered_normalized_counts.csv'))
hemato_sans_bm1 = hemato[hemato[:,2] != 'basal_bm1'] # Remove basal_bm1 (per paper's instructions)
hemato_sans_bm1_trim = hemato_sans_bm1[:,4:] # Get rid of headers
hemato_T = hemato_sans_bm1_trim.T # Reshape to get shape of (n_genes, n_cells)
hem_gene_vars = hemato_T.var(axis=1)
hem_gene_vars_ordered_idx = np.argsort(hem_gene_vars)[::-1]

# Get 7397 most varied
hem_genes_most_varied = hem_gene_vars_ordered_idx[:7397]
hemato_most_varied = hemato_T[hem_genes_most_varied]
np.savetxt('hemato_mostvar7397_unnorm.txt',hemato_most_varied, fmt='%i')

# Get 700 most varied
hem_genes_most_varied2 = hem_gene_vars_ordered_idx[:700]
hemato_most_varied2 = hemato_T[hem_genes_most_varied2]
np.savetxt('hemato_mostvar700_unnorm.txt',hemato_most_varied2, fmt='%i')
