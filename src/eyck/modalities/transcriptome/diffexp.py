import polyptich as pp
import scanpy as sc
import numpy as np

import pandas as pd
from eyck.modalities.transcriptome import symbol


def compare_two_groups(adata, grouping1, grouping2=None, **kwargs):
    
    if grouping2 is None:
        grouping2 = ~grouping1

    if grouping1.mean() == 1:
        raise ValueError("Grouping 1 is all true")
    if grouping1.mean() == 0:
        raise ValueError("Grouping 1 is all false")
    if grouping2.mean() == 0:
        raise ValueError("Grouping 2 is all false")
    
    adata.obs["oi"] = pd.Categorical(
        pp.utils.case_when(
            oi=grouping1,
            ref=grouping2,
        )
    )
    adata_test = adata[adata.obs["oi"].isin(["oi", "ref"])].copy()
    sc.tl.rank_genes_groups(
        adata_test, "oi", groups=["oi"], reference="ref", key_added="27vs6", **kwargs
    )
    diffexp = sc.get.rank_genes_groups_df(adata_test, "oi", key="27vs6").set_index(
        "names"
    )

    lfc = pd.Series(
        -np.array(adata_test.X[adata_test.obs["oi"].values == "ref"].mean(0)).flatten()
        + np.array(adata_test.X[adata_test.obs["oi"].values == "oi"].mean(0)).flatten(),
        index=adata_test.var.index,
    )
    if "symbol" in adata.var.columns:
        diffexp = diffexp.assign(symbol=lambda x: symbol(adata.var, x.index).values)
    diffexp.index.name = "gene"
    diffexp = diffexp.assign(
        lfc=lfc,
    )
    ##Get pct
    cellsGroup1 = adata_test.obs.loc[grouping1].index
    cellsGroup2 = adata_test.obs.loc[grouping2].index
    genesOI = diffexp.index.tolist()

    X1 = adata_test[cellsGroup1, genesOI].X[:10000]
    if not isinstance(X1, np.ndarray):
        X1 = X1.todense()
    X2 = adata_test[cellsGroup2, genesOI].X[:10000]
    if not isinstance(X2, np.ndarray):
        X2 = X2.todense()

    diffexp["pct.1"] = np.array(
        np.mean(X1 > 0, axis=0)
    )[0]
    diffexp["pct.2"] = np.array(
        np.mean(X2 > 0, axis=0)
    )[0]
    diffexp["scoreLM"] = np.where(
        diffexp["lfc"] >= 0,
        ((diffexp["pct.1"] + 0.01) / (diffexp["pct.2"] + 0.01)) * diffexp["lfc"],
        ((diffexp["pct.2"] + 0.01) / (diffexp["pct.1"] + 0.01)) * diffexp["lfc"],
    )
    diffexp = diffexp.sort_values("scoreLM", ascending=False)

    return diffexp


def compare_all_groups(adata, column):
    diffexp = pd.DataFrame()
    for group in adata.obs[column].cat.categories:
        diffexp_group = compare_two_groups(adata, adata.obs[column] == group)
        diffexp_group["group"] = group
        diffexp = pd.concat([diffexp, diffexp_group])
    return diffexp
