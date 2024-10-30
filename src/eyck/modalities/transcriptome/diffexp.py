import polyptich as pp
import scanpy as sc

import pandas as pd
from eyck.modalities.transcriptome import symbol
def compare_two_groups(adata, grouping1, grouping2 = None):
    if grouping2 is None:
        grouping2 = ~grouping1
    adata.obs["oi"] = pd.Categorical(pp.utils.case_when(
        oi=grouping1,
        ref=grouping2,
    ))
    adata_test = adata[adata.obs["oi"].isin(["oi", "ref"])].copy()
    sc.tl.rank_genes_groups(
        adata_test, "oi", groups=["oi"], reference="ref", key_added="27vs6"
    )
    diffexp = (
        sc.get.rank_genes_groups_df(adata_test, "oi", key="27vs6")
        .set_index("names")
        .assign(
            symbol=lambda x: symbol(adata.var, x.index).values
        )
    )
    return diffexp