import urllib.request
import tempfile
import pathlib
import eyck.modalities.transcriptome


def download_cellcycle_genes(path, check_exists=False):
    url = "https://raw.githubusercontent.com/scverse/scanpy_usage/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt"
    urllib.request.urlretrieve(url, path)
    return path


def get_cellcycle_genes(adata=None, path=None, organism="human"):
    if path is None:
        path = pathlib.Path(tempfile.NamedTemporaryFile().name)
        path = download_cellcycle_genes(path, check_exists=False)
    cell_cycle_genes = [x.strip() for x in open(path, "r")]

    s_genes = cell_cycle_genes[:43]
    if organism == "human":
        s_genes = [g for g in s_genes if g in adata.var["symbol"].tolist()]
        s_genes = eyck.modalities.transcriptome.gene_id(adata.var, s_genes)
    elif organism == "mouse":
        s_genes = [
            g.capitalize()
            for g in s_genes
            if g.capitalize() in adata.var["symbol"].tolist()
        ]
        s_genes = eyck.modalities.transcriptome.gene_id(adata.var, s_genes)
    g2m_genes = cell_cycle_genes[43:]
    if organism == "human":
        g2m_genes = [g for g in g2m_genes if g in adata.var["symbol"].tolist()]
        g2m_genes = eyck.modalities.transcriptome.gene_id(adata.var, g2m_genes)
    elif organism == "mouse":
        g2m_genes = [
            g.capitalize()
            for g in g2m_genes
            if g.capitalize() in adata.var["symbol"].tolist()
        ]
        g2m_genes = eyck.modalities.transcriptome.gene_id(adata.var, g2m_genes)
    return s_genes, g2m_genes
