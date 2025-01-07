import polyptich
import matplotlib as mpl
import pandas as pd
from .data import gene_id
from .data import Transcriptome
import scanpy as sc
import numpy as np
from typing import List
import datashader as ds
from datashader.mpl_ext import dsshow


def get_colors(n):
    if n <= 10:
        colors = mpl.colormaps["tab10"].colors[:n]
    elif n <= 20:
        colors = mpl.colormaps["tab20"].colors[:n]
    else:
        colors = mpl.colormaps["nipy_spectral"](np.linspace(0, 1, n))
    return [mpl.colors.to_hex(c) for c in colors]


def plot_embedding(
    transcriptome: Transcriptome,
    color: List[str] = None,
    panel_size: float = 2.0,
    cmap="magma",
    cmaps=None,
    palettes=None,
    norms=(0, "q.98"),
    colors="red",
    transforms=None,
    fig=None,
    grid=None,
    layer=None,
    embedding="X_umap",
    ax = None,
    cells_oi = None,
    datashader=None,
    ncol = 4,
    annotations = None,
    show_norm = True,
    size = None,
    title = None,
    legend = "on data",
    rasterized = False,
):
    """
    Plot cell-based features on the UMAP of the transcriptome.

    Parameters
    ----------
    transcriptome
        The transcriptome to plot. Can be a Transcriptome object or an AnnData object.
    color
        The features to plot.
    panel_size
        The size of the panel.
    cmap
        The colormap to use.
    palettes
        The palettes to use for categorical features. If None, a default palette will be used.
        If a dictionary, the keys should be the feature names and the values should be the palettes.
    colors
        The color to use for boolean features.
    grid
        The grid to add the panels to.
    embedding
        The embedding to plot.
    layer
        The layer to use.
    ax
        The axis to plot on.
    norms
        The normalization to use. If None, the default normalization will be used.
    legend
        Where to put the legend in case of categorical features. Can be "on data" or "on panel" or False/None.
    """
    if isinstance(transcriptome, sc.AnnData):
        transcriptome = Transcriptome.from_adata(adata=transcriptome)

    # if not isinstance(transcriptome, Transcriptome):
    #     raise ValueError("transcriptome must be a Transcriptome object")

    if ax is None:
        if grid is None:
            fig = polyptich.grid.Figure(polyptich.grid.Wrap(padding_width = 0.1, ncol = ncol))

            grid = fig.main
        else:
            fig = None

    if embedding not in transcriptome.adata.obsm:
        raise ValueError(f"Could not find embedding {embedding}")
    
    if cells_oi is None:
        cells_oi = np.ones(len(transcriptome.adata), dtype=bool)


    plotdata = plotdata_raw = pd.DataFrame(
        {
            "x": transcriptome.adata.obsm[embedding][cells_oi, 0],
            "y": transcriptome.adata.obsm[embedding][cells_oi, 1],
        }
    )

    if isinstance(color, str):
        color = [color]
    elif color is None:
        color = [transcriptome.obs.columns[0]]

    if annotations is None:
        annotations = {}

    if palettes is None:
        palettes = {}
    if norms is None:
        norms = {feature: (0, "q.95") for feature in color}
    elif isinstance(norms, tuple):
        norms = {feature: norms for feature in color}
    elif isinstance(norms, str):
        norms = {feature: norms for feature in color}
    elif not isinstance(norms, dict):
        norms = {feature: norms for feature in color}
    if transforms is None:
        transforms = {}
    if cmaps is None:
        cmaps = {}
    if isinstance(colors, str):
        colors = {g: colors for g in color}

    # determine width and height based on aspect ratio
    aspect = (plotdata["y"].max() - plotdata["y"].min()) / (
        plotdata["x"].max() - plotdata["x"].min()
    )
    panel_width = panel_size
    panel_height = panel_size * aspect

    cmap_default = cmap

    for feature in color:
        plotdata = plotdata_raw.copy()

        # determine feature and version
        version = "continuous"

        # expression
        if (feature in transcriptome.var.index) or (
            ("symbol" in transcriptome.var.columns)
            and (feature in transcriptome.var["symbol"].values)
        ):
            if feature not in transcriptome.var.index:
                label = feature
                gene = gene_id(transcriptome.var, feature, column="symbol")
            elif feature in transcriptome.var.index:
                gene = feature
                if "symbol" in transcriptome.var.columns:
                    label = transcriptome.var.loc[feature]["symbol"]
                else:
                    label = feature
            else:
                raise ValueError(f"Could not find gene {feature}")

            plotdata["z"] = sc.get.obs_df(
                transcriptome.adata[cells_oi], gene, layer=layer
            ).values

            if feature not in norms:
                q99 = plotdata["z"].quantile(0.999)
                if q99 == 0:
                    q99 = plotdata["z"].max()
                norms[feature] = mpl.colors.Normalize(0.0, q99)
            elif norms[feature] == "minmax":
                norms[feature] = mpl.colors.Normalize(
                    plotdata["z"].min(), plotdata["z"].max()
                )

        # obs feature
        elif feature in transcriptome.obs.columns:
            plotdata["z"] = transcriptome.obs.loc[cells_oi,feature].values

            if plotdata["z"].dtype == "category":
                plotdata["value"] = plotdata["z"]
                plotdata["z"] = plotdata["z"]
                version = "category"
            elif plotdata["z"].dtype == "object":
                plotdata["value"] = plotdata["z"]
                plotdata["z"] = plotdata["z"].astype("category")
                version = "category"
            elif plotdata["z"].dtype == "bool":
                version = "bool"
                plotdata["z"] = plotdata["z"].astype(int)
            else:
                plotdata["z"] = plotdata["z"]

            label = feature
        else:
            raise ValueError(f"Could not find feature {feature}")
        
        if not isinstance(norms, dict):
            norms = {feature: norms for feature in color}

        if feature not in norms:
            norms[feature] = (0, "q.98")

        if ax is None:
            current_ax = grid.add(polyptich.grid.Panel((panel_width, panel_height)))
        else:
            current_ax = ax
        if size is None:
            s = min(5, 10000 / len(plotdata))
        else:
            s = size
        scale = 20

        # check if datashader
        if (len(plotdata) < 10000) or (datashader is False):
            do_datashader = False
        else:
            do_datashader = True
        
        # actual plotting depending on version
        if version == "category":
            if feature in palettes:
                palette = palettes[feature]
                missing = set(plotdata["z"].cat.categories) - set(palette.index)
                if missing:
                    palette = palette.reindex(
                        plotdata["z"].cat.categories, fill_value="None"
                    )

                    palette[palette == "None"] = get_colors(len(missing))
            else:
                palette = pd.Series(
                    get_colors(len(plotdata["z"].cat.categories)),
                    index=plotdata["z"].cat.categories,
                )

            if do_datashader:
                dsshow(
                    plotdata,
                    ds.Point("x", "y"),
                    ds.count_cat("z"),
                    color_key=palette.to_dict(),
                    aspect="equal",
                    ax=current_ax,
                    plot_width=int(panel_width * 100),
                    plot_height=int(panel_height * 100),
                    height_scale=scale,
                    width_scale=scale,
                    alpha_range=[200, 255],
                )
            else:
                scatter = current_ax.scatter(
                    plotdata["x"],
                    plotdata["y"],
                    c=palette[plotdata["z"]],
                    s=s,
                    linewidths=0,
                    clip_on = False,
                )
                if rasterized:
                    scatter.set_rasterized(True)
                

        elif version == "continuous":
            # get cmap
            if feature in cmaps:
                cmap = cmaps[feature]
            else:
                cmap = cmap_default
            if isinstance(cmap, str):
                cmap = mpl.colormaps[cmap]

            # get norm
            if feature not in norms:
                norm = mpl.colors.Normalize()
            elif norms[feature] == "minmax":
                norm = mpl.colors.Normalize(
                    vmin=plotdata["z"].min(), vmax=plotdata["z"].max()
                )
            elif norms[feature] == "0max":
                norm = mpl.colors.Normalize(vmin=0, vmax=plotdata["z"].max())
            elif isinstance(norms[feature], str) and "q" in norms[feature]:
                qmin, qmax = norms[feature].split("q")[0]
            elif isinstance(norms[feature], tuple):
                if norms[feature][0] == "0":
                    zmin = 0
                elif isinstance(norms[feature][0], (int, float)):
                    zmin = norms[feature][0]
                elif norms[feature][0].startswith("q"):
                    zmin = np.quantile(plotdata["z"], float(norms[feature][0][1:]))
                elif norms[feature][0] == "min":
                    zmin = plotdata["z"].min()
                else:
                    zmin = norms[feature][0]
                    
                if norms[feature][1] == "0":
                    zmax = 0
                elif isinstance(norms[feature][1], (int, float)):
                    zmax = norms[feature][1]
                elif norms[feature][1].startswith("q"):
                    zmax = np.quantile(plotdata["z"], float(norms[feature][1][1:]))
                elif norms[feature][1] == "max":
                    zmax = plotdata["z"].max()
                else:
                    zmax = norms[feature][1]
                norm = mpl.colors.Normalize(zmin, zmax+1e-8)
            elif isinstance(norms[feature], mpl.colors.Normalize):
                norm = norms[feature]
            else:
                raise ValueError(f"Unknown normalization {norms[feature]}")

            # decide datashader or not
            if do_datashader:
                dsshow(
                    plotdata,
                    ds.Point("x", "y"),
                    ds.mean("z"),
                    cmap=cmap,
                    norm=norm,
                    aspect="equal",
                    ax=current_ax,
                    plot_width=int(panel_width * 100),
                    plot_height=int(panel_height * 100),
                    height_scale=scale,
                    width_scale=scale,
                    alpha_range=[200, 255],
                )
            else:
                plotdata = plotdata.sort_values("z")
                scatter = current_ax.scatter(
                    plotdata["x"],
                    plotdata["y"],
                    c=plotdata["z"].values,
                    s=s,
                    cmap=cmap,
                    norm = norm,
                    linewidths=0,
                    clip_on = False,
                )
                
                if rasterized:
                    scatter.set_rasterized(True)
                    
        elif version == "bool":
            if do_datashader:
                dsshow(
                    plotdata,
                    ds.Point("x", "y"),
                    ds.mean("z"),
                    cmap=["grey", colors[feature]],
                    norm=mpl.colors.Normalize(vmin=0, vmax=1),
                    aspect="equal",
                    ax=current_ax,
                    plot_width=int(panel_width * 100),
                    plot_height=int(panel_height * 100),
                    height_scale=scale,
                    width_scale=scale,
                    alpha_range=[200, 255],
                )
            else:
                plotdata = plotdata.sort_values("z")
                plotdata_1 = plotdata[plotdata["z"] == 1]
                plotdata_0 = plotdata[plotdata["z"] == 0]
                scatter = current_ax.scatter(
                    plotdata_0["x"],
                    plotdata_0["y"],
                    c="grey",
                    s=s,
                    linewidths=0,
                    clip_on = False,
                )
                if rasterized:
                    scatter.set_rasterized(True)
                scatter = current_ax.scatter(
                    plotdata_1["x"],
                    plotdata_1["y"],
                    c=colors[feature],
                    s=s,
                    linewidths=0,
                    clip_on = False,
                )
                if rasterized:
                    scatter.set_rasterized(True)

        current_ax.set_xlim(plotdata["x"].min(), plotdata["x"].max())
        current_ax.set_ylim(plotdata["y"].min(), plotdata["y"].max())
        current_ax.spines["top"].set_visible(False)
        current_ax.spines["right"].set_visible(False)
        current_ax.spines["bottom"].set_visible(False)
        current_ax.spines["left"].set_visible(False)
        current_ax.set_xticks([])
        current_ax.set_yticks([])

        if title is None:
            current_title = label
        else:
            if isinstance(title, str):
                current_title = title
            elif isinstance(title, dict):
                current_title = title.get(feature, label)
            # check if function
            elif callable(title):
                current_title = title(feature)
            else:
                raise ValueError("title must be a string or a dictionary")
        if feature in annotations:
            current_title = label + "\n" + annotations[feature]

        current_ax.set_title(current_title, fontsize = 10)

        if (version == "category"):
            if legend == "on data":
                texts = []
                plotdata_grouped = plotdata.groupby("value", observed=True)[
                    ["x", "y"]
                ].median()
                for i, row in plotdata_grouped.iterrows():
                    text = current_ax.text(row["x"], row["y"], i, fontsize=8, color=palette[i], fontweight="bold")
                    text.set_path_effects(
                        [
                            mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFFAA"),
                            mpl.patheffects.Stroke(linewidth=1, foreground="#333333"),
                            mpl.patheffects.Normal(),
                        ]
                    )
                    texts.append(text)

                try:
                    import adjustText

                    adjustText.adjust_text(
                        texts, arrowprops=dict(arrowstyle="->", color="black"), ax=current_ax
                    )

                    # fig.plot_hooks.append(
                    #     lambda:
                    # )
                except ImportError:
                    pass

        if version == "continuous" and show_norm:
            if (norm.vmin is not None) and (norm.vmax is not None):
                text = current_ax.text(
                    0.0,
                    0.0,
                    f"[{norm.vmin:.1f}, {norm.vmax:.1f}]",
                    fontsize=6,
                    color="grey",
                    transform=current_ax.transAxes,
                )
                text.set_path_effects(
                    [
                        mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFF"),
                        mpl.patheffects.Normal(),
                    ]
                )

    return fig


def plot_umap(
    transcriptome: Transcriptome,
    color: List[str] = None,
    panel_size: float = 1.5,
    cmap="magma",
    cmaps=None,
    palettes=None,
    norms=(0, "q.98"),
    colors="tomato",
    transforms=None,
    fig=None,
    grid=None,
    layer=None,
    ax = None,
    cells_oi = None,
    datashader = None,
    ncol = 4,
    annotations = None,
    **kwargs,
) -> polyptich.grid.Figure:
    return plot_embedding(
        transcriptome=transcriptome,
        color=color,
        panel_size=panel_size,
        cmap=cmap,
        cmaps=cmaps,
        palettes=palettes,
        norms=norms,
        colors=colors,
        transforms=transforms,
        fig=fig,
        grid=grid,
        layer = layer,
        embedding = "X_umap",
        ax = ax,
        cells_oi = cells_oi,
        datashader = datashader,
        ncol = ncol,
        annotations = annotations,
        **kwargs
    )


def plot_umap_categories(
    transcriptome: Transcriptome,
    feature: str,
    panel_size: float = 2.0,
    colors="red",
    fig=None,
    grid=None,
    labels = None,
    **kwargs,
) -> polyptich.grid.Figure:
    if grid is None:
        fig = polyptich.grid.Figure(polyptich.grid.Wrap())
        grid = fig.main
    for category in transcriptome.obs[feature].cat.categories:
        transcriptome.obs[feature + "_" + category] = (
            transcriptome.obs[feature] == category
        )
        if labels is None:
            title = category
        else:
            title = labels[category]
        plot_umap(
            transcriptome=transcriptome,
            color=[feature + "_" + category],
            panel_size=panel_size,
            colors=colors,
            grid=grid,
            title = title,
            **kwargs
        )
    return fig




def plot_umap_categorized(
    transcriptome: Transcriptome,
    feature: str,
    color,
    panel_size: float = 2.0,
    fig=None,
    grid=None,
    **kwargs,
) -> polyptich.grid.Figure:
    if grid is None:
        fig = polyptich.grid.Figure(polyptich.grid.Wrap())
        grid = fig.main
    for category in transcriptome.obs[feature].cat.categories:
        transcriptome.obs[feature + "_" + category] = (
            transcriptome.obs[feature] == category
        )
        ax = grid.add(polyptich.grid.Panel((panel_size, panel_size)))
        plot_umap(
            transcriptome=transcriptome,
            cells_oi = transcriptome.obs[feature + "_" + category],
            color=color,
            panel_size=panel_size,
            ax = ax,
            title = category,
            **kwargs
        )
    return fig




def create_continuous_colorbar(ax, norm = mpl.colors.Normalize(0, 1), cmap = None):
    if cmap is None:
        cmap = mpl.cm.magma
    mappable = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    import matplotlib.pyplot as plt
    colorbar = plt.colorbar(
        mappable, cax=ax, orientation="vertical", extend="max"
    )
    colorbar.set_label("Expression")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["0", "Q95"])