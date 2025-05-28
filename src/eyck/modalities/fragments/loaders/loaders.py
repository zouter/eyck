from __future__ import annotations

from typing import Union

import numpy as np
import jax.numpy as jnp
import jax
import functools
import eyck.modalities

# try to load the shared library
# typically, this will be installed as a python extension
try:
    from . import helpers  # pylint: disable=C0413,E0611
# however, during developement, we want to load the cython source directly
except ImportError:
    import pyximport

    pyximport.install(
        reload_support=True,
        language_level=3,
        setup_args=dict(include_dirs=[np.get_include()]),
        build_in_temp=False,
    )
    from . import helpers  # pylint: disable=C0413,E0611

import dataclasses

from latenta.utils import isinstance2 as isinstance

import eyck.modalities.fragments

def open_memmap(path):
    import pickle

    filename = path.with_suffix(".dat")
    filename_meta = path.with_suffix(".meta")
    metadata = pickle.load(open(filename_meta, "rb"))
    return np.memmap(
        filename, dtype=metadata["dtype"], shape=metadata["shape"], mode="r"
    )


class Fragments():
    """
    Basic loader for fragments. This requires either `regionxcell_indptr` (for a Fragments) or `regionxcell_fragmentixs_indptr` (for a FragmentsView) to be present.

    Example:
        ```
        loader = Fragments(fragments, cellxregion_batch_size=1000)
        minibatch = {"cell":np.arange(100), "region":np.arange(100)}
        data = loader.load(minibatch)
        data.coordinates
        ```
    """

    cellxregion_batch_size: int

    preloaded = False

    out_coordinates: jnp.ndarray
    out_regionmapping: jnp.ndarray
    out_local_cellxregion_ix: jnp.ndarray

    n_regions: int
    is_view: bool

    def __init__(
        self,
        fragments: Union[
            eyck.modalities.fragments.Fragments,
            eyck.modalities.fragments.FragmentsView,
        ],
        cell_batch_size:int = None,
        region_batch_size:int = None,
        n_fragment_per_cellxregion: int = None,
        buffer_size_multiplier=1.5,  # increase this if crashing
        requests = tuple(["coordinates", "local_cellxregion_ix"]),
    ):
        """
        Parameters:
            fragments: Fragments object
            cellxregion_batch_size: maximum number of cell x region combinations that will be loaded
            n_fragment_per_cellxregion: estimated number of the number of fragments per cell x region combination, used for preallocation
        """

        if cell_batch_size is None:
            cell_batch_size = len(fragments.obs)
        if region_batch_size is None:
            region_batch_size = len(fragments.regions)

        self.cellxregion_batch_size = cell_batch_size * region_batch_size

        # store auxilliary information
        window = fragments.regions.window
        self.window = window

        # create buffers for coordinates
        if n_fragment_per_cellxregion is None:
            n_fragment_per_cellxregion = fragments.estimate_fragment_per_cellxregion()
        fragment_buffer_size = int(
            n_fragment_per_cellxregion * self.cellxregion_batch_size * buffer_size_multiplier
        )
        self.fragment_buffer_size = fragment_buffer_size

        self.n_regions = fragments.n_regions

        # set up readers and determine if we are dealing with a view or not
        if isinstance(fragments, eyck.modalities.fragments.Fragments):
            self.regionxcell_indptr_reader = fragments.regionxcell_indptr.open_reader(
                {"context": {"data_copy_concurrency": {"limit": 1}}}
            )
            self.coordinates_reader = fragments.coordinates.open_reader(
                {
                    "context": {
                        "data_copy_concurrency": {"limit": 1},
                    }
                }
            )
            self.is_view = False

        elif isinstance(fragments, eyck.modalities.fragments.view.FragmentsView):
            self.regionxcell_indptr_reader = fragments.regionxcell_indptr.open_reader()
            self.coordinates_reader = fragments.coordinates.open_reader()

            if "strand" in fragments.regions.coordinates.columns:
                self.region_strands = fragments.regions.coordinates["strand"].values
            else:
                self.region_strands = np.ones(
                    (len(fragments.regions.coordinates),), dtype=np.int8
                )
            self.region_centers = (
                fragments.regions.coordinates["start"] - fragments.regions.window[0]
            ).values * (self.region_strands == 1).astype(int) + (
                fragments.regions.coordinates["end"] + fragments.regions.window[0]
            ).values * (
                self.region_strands == -1
            ).astype(
                int
            )
            self.is_view = True
        else:
            raise ValueError(
                "fragments must be either a Fragments or FragmentsView object",
                type(fragments),
            )

        self.n_cells = fragments.n_cells

        self.requests = set()
        self.add_requests(set(requests))

        if "celltype" in fragments.obs.columns:
            self.celltypes = fragments.obs["celltype"].cat.codes.values

    def preload(self):
        self.out_fragmentixs = np.zeros((self.fragment_buffer_size,), dtype=np.int64)
        self.out_local_cellxregion_ix = np.zeros(
            (self.fragment_buffer_size,), dtype=np.int64
        )

        self.preloaded = True

    def load(self, minibatch) -> dict:
        """
        Load a minibatch of fragments.

        Parameters:
            minibatch: Minibatch object

        Returns:
            The loaded fragments
        """
        if not self.preloaded:
            self.preload()

        if "region" not in minibatch:
            minibatch["region"] = np.arange(self.n_regions)
        if "cell" not in minibatch:
            minibatch["cell"] = np.arange(self.n_cells)

        if (
            len(minibatch["cell"]) * len(minibatch["region"])
        ) > self.cellxregion_batch_size:
            raise ValueError(
                "Too many cell x region requested, increase cellxregion_batch_size at loader initialization"
            )

        if self.is_view:
            # load the fragment indices using pointers to the regionxcell fragment indices
            regionxcell_ixs = (
                minibatch["region"] * self.n_cells + minibatch["cell"][:, None]
            ).flatten()
            n_fragments = helpers.multiple_arange(
                np.array(self.regionxcell_indptr_reader[regionxcell_ixs, 0]),
                np.array(self.regionxcell_indptr_reader[regionxcell_ixs, 1]),
                self.out_fragmentixs,
                self.out_local_cellxregion_ix,
            )

            assert (
                n_fragments < self.fragment_buffer_size
            ), "fragment buffer size too small"

            regionxcell_fragmentixs = self.out_fragmentixs[:n_fragments]
            local_cellxregion_ix = self.out_local_cellxregion_ix[:n_fragments]

            regionmapping = minibatch["region"][
                local_cellxregion_ix % len(minibatch["region"])
            ]
            coordinates = self.coordinates_reader[
                regionxcell_fragmentixs
            ]  # this is typically the slowest part by far

            # center coordinates around region centers, flip based on strandedness
            coordinates = (
                coordinates - self.region_centers[regionmapping][:, None]
            ) * self.region_strands[regionmapping][:, None]
        else:
            regionxcell_ixs = (
                minibatch["region"] * self.n_cells + minibatch["cell"][:, None]
            ).flatten()
            n_fragments = helpers.multiple_arange(
                self.regionxcell_indptr_reader[regionxcell_ixs],
                self.regionxcell_indptr_reader[regionxcell_ixs + 1],
                self.out_fragmentixs,
                self.out_local_cellxregion_ix,
            )
            regionxcell_fragmentixs = np.resize(self.out_fragmentixs, n_fragments)
            coordinates = self.coordinates_reader[
                regionxcell_fragmentixs
            ]  # this is typically the slowest part by far
            local_cellxregion_ix = np.resize(self.out_local_cellxregion_ix, n_fragments)
            regionmapping = minibatch["region"][
                local_cellxregion_ix % len(minibatch["region"])
            ]

        # pad
        pad_n = self.fragment_buffer_size - n_fragments
        coordinates = np.pad(
            coordinates,
            ((0, pad_n), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        local_cellxregion_ix = np.pad(
            local_cellxregion_ix, (0, pad_n), mode="constant", constant_values=0
        )

        mask = np.zeros((self.fragment_buffer_size,), dtype=bool)
        mask[-pad_n:] = True

        result = {}
        result["coordinates"] = coordinates
        result["mask"] = mask

        # libsize
        if "libsize" in self.requests:
            result["libsize"] = self.library_size[minibatch["cell"]]

        if "regionmapping" in self.requests:
            regionmapping = minibatch["region"][
                local_cellxregion_ix % len(minibatch["region"])
            ]
            result["regionmapping"] = regionmapping

        if "local_indices" in self.requests:
            result["local_indices"] = np.stack([
                local_cellxregion_ix // len(minibatch["region"]),
                local_cellxregion_ix % len(minibatch["region"]),
            ], axis=1)

        if "mask" in self.requests:
            result["mask"] = mask

        if "cut_coordinates" in self.requests:
            cut_coordinates = coordinates.flatten()
            selected = (cut_coordinates >= self.window[0]) & (cut_coordinates < self.window[1])
            cut_coordinates = cut_coordinates[selected] - self.window[0]

            local_cut_cellxregion_ix = local_cellxregion_ix.repeat(
                2, -1
            ).T.flatten()[selected]
            local_cut_cell_ix = np.floor_divide(
                local_cut_cellxregion_ix, self.n_regions
            )

            pad_n = self.fragment_buffer_size*2 - len(cut_coordinates)
            cut_coordinates = np.pad(
                cut_coordinates,
                ((0, pad_n)),
                mode="constant",
                constant_values=0,
            )

            result["cut_coordinates"] = cut_coordinates

            if "cut_region_cluster" in self.requests:
                cut_region = np.repeat(minibatch["region"][
                    local_cellxregion_ix % len(minibatch["region"])
                ], 2, -1).T.flatten()[selected]

                cut_cluster = self.celltypes[minibatch["cell"]][local_cut_cell_ix]

                cut_region_cluster = np.stack([cut_region, cut_cluster], axis=-1)

                cut_region_cluster = np.pad(
                    cut_region_cluster,
                    ((0, pad_n), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                result["cut_region_cluster"] = cut_region_cluster

            if "cut_mask" in self.requests:
                cut_mask = np.zeros((self.fragment_buffer_size*2,), dtype=bool)
                cut_mask[-pad_n:] = True
                result["cut_mask"] = cut_mask

        return result

    def add_requests(self, requests):
        for request in requests:
            if request == "libsize":
                if "libsize" not in self.requests:
                    library_size = self.fragments.libsize
                    self.library_size = ((library_size - library_size.mean()) / library_size.std()
                    ).float()
                    self.requests.add("libsize")
            elif request == "regionmapping":
                self.requests.add("regionmapping")
            elif request == "coordinates":
                self.requests.add("coordinates")
            elif request == "local_cellxregion_ix":
                self.requests.add("local_cellxregion_ix")
            elif request == "local_indices":
                self.requests.add("local_indices")
            elif request == "mask":
                self.requests.add("mask")
            elif request == "cut_coordinates":
                self.requests.add("cut_coordinates")
            elif request == "cut_mask":
                self.requests.add("cut_mask")
            elif request == "cut_region_cluster":
                self.requests.add("cut_region_cluster")
            else:
                raise ValueError("Request not supported ", request)
    


@dataclasses.dataclass
class CutsResult:
    coordinates: np.ndarray
    local_cellxregion_ix: np.ndarray
    n_regions: int
    n_fragments: int
    n_cuts: int
    window: np.ndarray

    @property
    def local_region_ix(self):
        return self.local_cellxregion_ix % self.n_regions

    @property
    def local_cell_ix(self):
        return np.div(
            self.local_cellxregion_ix, self.n_regions, rounding_mode="floor"
        )

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_cellxregion_ix = self.local_cellxregion_ix.to(device)
        return self


class Cuts(Fragments):
    def load(self, minibatch) -> CutsResult:
        """
        Load a minibatch of cuts.

        Parameters:
            minibatch: Minibatch object

        Returns:
            The loaded cut sites
        """
        result = super().load(minibatch)

        cut_coordinates = result.coordinates.flatten()

        n_cuts_per_fragment = result.coordinates.shape[1]
        local_cellxregion_ix = result.local_cellxregion_ix.expand(
            n_cuts_per_fragment, -1
        ).T.flatten()

        # selected = np.random.rand(len(cut_coordinates)) < 0.2
        # cut_coordinates = cut_coordinates[selected]
        # local_cellxregion_ix = local_cellxregion_ix[selected]

        return CutsResult(
            coordinates=cut_coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            n_regions=len(minibatch["region"]),
            n_fragments=result.n_fragments,
            n_cuts=result.n_fragments * n_cuts_per_fragment,
            window=self.window,
        )
