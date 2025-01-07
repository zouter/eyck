import polyptich as pp
import scanpy as sc
import numpy as np

import pandas as pd
from eyck.modalities.transcriptome import symbol
def compare_two_groups(adata, grouping1, grouping2 = None, **kwargs):
    if grouping2 is None:
        grouping2 = ~grouping1
    adata.obs["oi"] = pd.Categorical(pp.utils.case_when(
        oi=grouping1,
        ref=grouping2,
    ))
    adata_test = adata[adata.obs["oi"].isin(["oi", "ref"])].copy()
    sc.tl.rank_genes_groups(
        adata_test, "oi", groups=["oi"], reference="ref", key_added="27vs6", **kwargs
    )
    diffexp = (
        sc.get.rank_genes_groups_df(adata_test, "oi", key="27vs6")
        .set_index("names")
    )

    lfc = pd.Series(-np.array(adata_test.X[adata_test.obs["oi"] == "ref"].mean(0)).flatten() + np.array(adata_test.X[adata_test.obs["oi"] == "oi"].mean(0)).flatten(), index = adata_test.var.index)
    if "symbol" in adata.var.columns:
        diffexp = diffexp.assign(
            symbol=lambda x: symbol(adata.var, x.index).values
        )
    diffexp.index.name = "gene"
    diffexp = diffexp.assign(
        lfc=lfc,
    )
    return diffexp

def compare_all_groups(adata, column):
    diffexp = pd.DataFrame()
    for group in adata.obs[column].cat.categories:
        diffexp_group = compare_two_groups(adata, adata.obs[column] != group)
        diffexp_group["group"] = group
        diffexp = pd.concat([diffexp, diffexp_group])
    return diffexp
