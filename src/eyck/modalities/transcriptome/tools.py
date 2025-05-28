import numpy as np
import pandas as pd
import scanpy as sc


def paircorr(x, y, dim=-2):
    divisor = y.std(dim, keepdims=True) * x.std(dim, keepdims=True)
    divisor[np.isclose(divisor, 0)] = 1.0
    cor = (x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))  # / divisor
    return cor

def refcor(x, y, dim=0, eps=0.000001, n=1000):
    single = False
    if x.ndim == 1:
        x = x[:, None]
        single = True
    x = x[:n]
    y = y[:n]
    divisor = (y.std(0)[None, :] * x.std(0)[:, None]) + eps
    numerator = ((x - x.mean())[:, :, None] * (y - y.mean(0))[:, None, :]).mean(0)
    cor = numerator / divisor
    if single:
        return cor[0]
    return cor


def correlate_single_gene(adata, gene_id, max_n = 2000):
    gene_ix = adata.var.index.get_loc(gene_id)
    try:
        X = np.array(adata.X[:max_n, :].todense())
    except AttributeError:
        X = adata.X[:max_n, :]
    x = X[:, gene_ix]
    y = np.array(X[:max_n, :])
    return pd.Series(refcor(x, y, n = max_n), adata.var.index).sort_values(ascending=False)



def correlate_feature(adata, feature, max_n = 2000):
    x = sc.get.obs_df(adata, keys=[feature]).values.flatten()[:max_n]
    try:
        X = np.array(adata.X[:max_n, :].todense())
    except AttributeError:
        X = adata.X[:max_n, :]
    y = np.array(X[:max_n, :])
    return pd.Series(refcor(x, y, n = max_n), adata.var.index).sort_values(ascending=False)


def jaccard_single_gene(adata, gene_id, max_n = 2000):
    gene_ix = adata.var.index.get_loc(gene_id)
    try:
        X = np.array(adata.X[:max_n, :].todense())
    except AttributeError:
        X = adata.X[:max_n, :]
    x = X[:, gene_ix]
    y = np.array(X[:max_n, :])
    a = x > 0
    b = y > 0

    y = (a[:, None] & b).sum(0) / (a[:, None] | b).sum(0)
    return pd.Series(y , adata.var.index).sort_values(ascending=False)



def reorder_umap(adata, ordering):
    """
    Flip UMAP x and y coordinates so that the cells with the lowest numbers or on top left
    """
    cor_x = np.corrcoef(adata.obsm["X_umap"][:, 0], ordering)[0, 1]
    cor_y = np.corrcoef(adata.obsm["X_umap"][:, 1], ordering)[0, 1]
    print(cor_x, cor_y)
    if cor_x > 0:
        adata.obsm["X_umap"][:, 0] = -adata.obsm["X_umap"][:, 0]
    if cor_y > 0:
        adata.obsm["X_umap"][:, 1] = -adata.obsm["X_umap"][:, 1]
    return adata
