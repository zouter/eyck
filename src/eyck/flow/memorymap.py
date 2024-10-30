import copy
import os

import numpy as np
import shutil
import pickle
import scipy

from .objects import Obj

from .objects import format_size, get_size

default_spec_create = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
    "metadata": {},
}

default_spec_write = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
}


default_spec_read = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
    "open": True,
}


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


class Memorymap(Obj):
    """
    A multidimensional array stored on disk using memory maps
    """

    def __init__(
        self,
        spec_create: dict = default_spec_create,
        spec_write: dict = default_spec_write,
        spec_read: dict = default_spec_read,
        dtype=None,
        name=None,
        compression="blosc",
        chunks=None,
        shape=(0,),
    ):
        self.name = name
        self.spec_create = copy.deepcopy(spec_create)
        self.spec_write = copy.deepcopy(spec_write)
        self.spec_read = copy.deepcopy(spec_read)

        if dtype is not None:
            self.spec_create["metadata"]["dtype"] = dtype

        if compression is not None:
            if compression == "blosc":
                self.spec_create["metadata"]["compressor"] = {
                    "id": "blosc",
                    "clevel": 3,
                    "cname": "zstd",
                    "shuffle": 2,
                }
        else:
            self.spec_create["metadata"]["compressor"] = None

        if chunks is not None:
            self.spec_create["metadata"]["chunks"] = chunks

        if shape is not None:
            self.spec_create["metadata"]["shape"] = shape

        if "dtype" not in self.spec_create["metadata"]:
            raise ValueError("dtype must be specified")

    def get_path(self, folder):
        return folder / (self.name + ".dat")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + self.name
            if not hasattr(obj, name):
                instance = MemorymapInstance(
                    self.name,
                    self.get_path(obj.path),
                    self.spec_create,
                    self.spec_write,
                    self.spec_read,
                )
                setattr(obj, name, instance)
            return getattr(obj, name)

    def __set__(self, obj, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Must be an numpy ndarray, not " + str(type(value)))
        path = self.get_path(obj.path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.__get__(obj)[:] = value

    def exists(self, obj):
        return self.__get__(obj).exists()

    def _repr_html_(self, obj):
        instance = self.__get__(obj)
        return instance._repr_html_()


class MemorymapInstance:
    _obj = None

    _fixed_reader = None

    def __init__(self, name, path, spec_create, spec_write, spec_read):
        self.name = name
        self.path = path
        self.spec_create = deep_update(
            copy.deepcopy(spec_create), {"kvstore": {"path": str(path)}}
        )
        self.spec_write = deep_update(
            copy.deepcopy(spec_write), {"kvstore": {"path": str(path)}}
        )
        self.spec_read = deep_update(
            copy.deepcopy(spec_read), {"kvstore": {"path": str(path)}}
        )

    @property
    def path_metadata(self):
        return self.path.with_suffix(".meta")

    def open_reader(self, spec=None, old=False):
        if self._fixed_reader is not None:
            return self._fixed_reader
        if spec is None:
            spec = self.spec_read
        else:
            spec = deep_update(copy.deepcopy(self.spec_read), spec)

        metadata = self.open_metadata()
        path = self.path
        if old:
            path = path.with_suffix(".dat.old")

        if not str(path).startswith("memory"):
            fp = np.memmap(
                path, dtype=metadata["dtype"], mode="r", shape=tuple(metadata["shape"])
            )
        else:
            fp = self._obj
        return fp

    def fix_reader(self):
        if self._fixed_reader is None:
            self._fixed_reader = self.open_reader()

    def open_writer(self, spec=None):
        if spec is None:
            spec = self.spec_write
        else:
            spec = deep_update(copy.deepcopy(self.spec_write), spec)
        metadata = self.open_metadata()
        if not str(self.path).startswith("memory"):
            fp = np.memmap(
                self.path, dtype=metadata["dtype"], mode="r+", shape=metadata["shape"]
            )
        else:
            fp = self._obj
        pickle.dump(metadata, self.path_metadata.open("wb"))
        return fp

    def create_metadata(self, spec, shape, dtype):
        if "metadata" not in spec:
            spec["metadata"] = {}
        metadata = spec["metadata"]

        if shape is not None:
            metadata["shape"] = shape

        if dtype is not None:
            metadata["dtype"] = dtype
        assert "dtype" in metadata
        assert "shape" in metadata
        return metadata

    def open_creator(self, spec=None, shape=None, dtype=None):
        if spec is None:
            spec = self.spec_create
        else:
            spec = deep_update(copy.deepcopy(self.spec_create), spec)

        metadata = self.create_metadata(spec, shape, dtype)

        if self.path.exists():
            self.path.unlink()
        if self.path_metadata.exists():
            self.path_metadata.unlink()

        if np.prod(metadata["shape"]) == 0:
            return None

        if not str(self.path).startswith("memory"):
            fp = np.memmap(
                self.path,
                dtype=metadata["dtype"],
                mode="w+",
                shape=tuple(metadata["shape"]),
            )
        else:
            self._obj = np.zeros(metadata["shape"], dtype=metadata["dtype"])
            fp = self._obj
        pickle.dump(metadata, self.path_metadata.open("wb"))
        return fp

    def exists(self):
        return (self._obj is not None) or (
            self.path.exists() and self.path_metadata.exists()
        )

    @property
    def info(self):
        return {
            "disk_space": get_size(self.path),
        }

    def __getitem__(self, key):
        return self.open_reader()[key]

    def __setitem__(self, key, value):
        if not self.exists():
            writer = self.open_creator(shape=value.shape, dtype=value.dtype.name)
        else:
            writer = self.open_writer()

        writer[key] = value

    def open_metadata(self):
        return pickle.load(self.path_metadata.open("rb"))

    @property
    def shape(self):
        return self.open_metadata()["shape"]

    def __len__(self):
        return self.shape[0]

    def _repr_html_(self):
        shape = "[" + ",".join(str(x) for x in self.shape) + "]"
        if not str(self.path).startswith("memory"):
            size = format_size(get_size(self.path))
        else:
            size = ""
        return f"<span class='iconify' data-icon='mdi-axis-arrow'></span> <b>{self.name}</b> {shape}, {size}"

    def extend(self, value):
        if not self.exists():
            writer = self.open_creator(dtype=value.dtype, shape=value.shape)
            writer[:] = value
        else:
            if not str(self.path).startswith("memory"):
                writer = self.open_writer()
                assert len(value.shape) == len(self.shape)
                assert value.shape[1:] == self.shape[1:]

                self.path.rename(self.path.with_suffix(".dat.old"))

                reader = self.open_reader(old=True)
                metadata = self.open_metadata()
                metadata["shape"] = (
                    metadata["shape"][0] + value.shape[0],
                    *metadata["shape"][1:],
                )

                writer = self.open_creator(
                    dtype=metadata["dtype"], shape=metadata["shape"]
                )

                writer[: writer.shape[0] - value.shape[0]] = reader[:]
                writer[writer.shape[0] - value.shape[0] :] = value

                self.path.with_suffix(".dat.old").unlink()

                pickle.dump(metadata, self.path_metadata.open("wb"))
            else:
                self._obj = np.concatenate([self._obj, value], axis=0)
                metadata = self.open_metadata()
                metadata["shape"] = self._obj.shape
                pickle.dump(metadata, self.path_metadata.open("wb"))


class CSRMemorymap(Obj):
    """
    A 2D CSR array stored on disk using memory maps
    """

    def __init__(
        self,
        spec_create: dict = default_spec_create,
        spec_write: dict = default_spec_write,
        spec_read: dict = default_spec_read,
        dtype=None,
        name=None,
        compression="blosc",
        chunks=None,
        shape=(0,),
    ):
        self.name = name
        self.spec_create = copy.deepcopy(spec_create)
        self.spec_write = copy.deepcopy(spec_write)
        self.spec_read = copy.deepcopy(spec_read)

        if dtype is not None:
            self.spec_create["metadata"]["dtype"] = dtype

        if shape is not None:
            self.spec_create["metadata"]["shape"] = shape

        if "dtype" not in self.spec_create["metadata"]:
            raise ValueError("dtype must be specified")

    def get_path(self, folder):
        return folder / (self.name + ".dat")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + self.name
            if not hasattr(obj, name):
                instance = CSRMemorymapInstance(
                    self.name,
                    self.get_path(obj.path),
                    self.spec_create,
                    self.spec_write,
                    self.spec_read,
                )
                setattr(obj, name, instance)
            return getattr(obj, name)

    def __set__(self, obj, value):
        if not isinstance(value, scipy.sparse._csr.csr_matrix):
            raise ValueError("Must be an scipy csr matrix, not " + str(type(value)))
        path = self.get_path(obj.path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)

        self.__get__(obj).write(value)

    def exists(self, obj):
        return self.__get__(obj).exists()

    def _repr_html_(self, obj):
        instance = self.__get__(obj)
        return instance._repr_html_()


class CSRMemorymapInstance(MemorymapInstance):
    def open_reader(self, spec=None, old=False):
        if self._fixed_reader is not None:
            return self._fixed_reader
        if spec is None:
            spec = self.spec_read
        else:
            spec = deep_update(copy.deepcopy(self.spec_read), spec)

        metadata = self.open_metadata()
        path = self.path
        if old:
            path = path.with_suffix(".dat.old")

        if not str(path).startswith("memory"):
            if "preload_dense" in spec and spec["preload_dense"]:
                fp = CSRMemorymapReader(path, metadata["dtype"], metadata["shape"])
                import scipy.sparse

                csr = scipy.sparse.csr_matrix(
                    (fp.data[:], fp.indices[:], fp.indptr[:]), shape=metadata["shape"]
                )
                fp = csr.todense()
            else:
                fp = CSRMemorymapReader(path, metadata["dtype"], metadata["shape"])
        else:
            fp = self._obj
        return fp

    def fix_reader(self):
        if self._fixed_reader is None:
            self._fixed_reader = self.open_reader()

    def write(self, value):
        if str(self.path).startswith("memory"):
            self._obj = value
            self._metadata = self.create_metadata(
                {}, dtype=value.dtype, shape=value.shape
            )
        else:
            fp_data = np.lib.format.open_memmap(
                self.path.with_suffix(".data"),
                dtype=value.data.dtype,
                mode="w+",
                shape=value.data.shape,
            )
            fp_indices = np.lib.format.open_memmap(
                self.path.with_suffix(".indices"),
                dtype=np.int32,
                mode="w+",
                shape=value.indices.shape,
            )
            fp_indptr = np.lib.format.open_memmap(
                self.path.with_suffix(".indptr"),
                dtype=np.int64,
                mode="w+",
                shape=value.indptr.shape,
            )

            fp_data[:] = value.data
            fp_indices[:] = value.indices
            fp_indptr[:] = value.indptr

            metadata = self.create_metadata({}, dtype=value.dtype, shape=value.shape)
            pickle.dump(metadata, self.path_metadata.open("wb"))

    def exists(self):
        return (self._obj is not None) or (
            self.path.exists() and self.path_metadata.exists()
        )

    @property
    def info(self):
        return {
            "disk_space": get_size(self.path),
        }

    def __getitem__(self, key):
        reader = self.open_reader()
        return reader[key]

    def __setitem__(self, key, value):
        assert key == slice(None)
        self.write(value)

    @property
    def data(self):
        return self.open_reader().data

    @property
    def indptr(self):
        return self.open_reader().indptr

    @property
    def indices(self):
        return self.open_reader().indices

    def sum(self, axis=1):
        if axis == 1:
            return np.bincount(
                np.repeat(np.arange(len(self.indptr) - 1), np.diff(self.indptr)),
                weights=self.data,
                minlength=self.shape[0],
            )
        elif axis == 0:
            return np.bincount(self.indices, weights=self.data, minlength=self.shape[1])
        else:
            raise NotImplementedError()


def indptr_to_indices(x):
    n = len(x) - 1
    return np.repeat(np.arange(n), np.diff(x))


class CSRMemorymapReader:
    def __init__(self, path, dtype, shape):
        self.path = path
        self.indices = np.lib.format.open_memmap(path.with_suffix(".indices"), mode="r")
        self.indptr = np.lib.format.open_memmap(path.with_suffix(".indptr"), mode="r")
        self.data = np.lib.format.open_memmap(path.with_suffix(".data"), mode="r")
        self.shape = shape

    def __getitem__(self, x):
        if isinstance(x, slice) and (x == slice(None)):
            return np.array(scipy.sparse.csr_matrix(
                (self.data, self.indices, self.indptr), shape=self.shape
            ).todense())
        elif isinstance(x, tuple):
            print(x)
            raise NotImplementedError()
        elif isinstance(x, np.ndarray):
            indptr_start = self.indptr[x]
            indptr_end = self.indptr[x + 1]
            indices_row = np.repeat(np.arange(len(x)), indptr_end - indptr_start)
            indices_col = np.concatenate(
                [
                    self.indices[start:end]
                    for start, end in zip(indptr_start, indptr_end)
                ]
            )
            data = np.concatenate(
                [self.data[start:end] for start, end in zip(indptr_start, indptr_end)]
            )

            y = np.zeros([len(x), self.shape[1]], dtype=self.data.dtype)
            y[indices_row, indices_col] = data

            return y
        else:
            raise NotImplementedError(type(x))


class Memorymaps(Obj):
    """
    A multidimensional array stored on disk using memory maps
    """

    def __init__(
        self,
        spec_create: dict = default_spec_create,
        spec_write: dict = default_spec_write,
        spec_read: dict = default_spec_read,
        dtype=None,
        name=None,
        compression="blosc",
        chunks=None,
        shape=(0,),
        value=None,
    ):
        self.name = name
        self.spec_create = copy.deepcopy(spec_create)
        self.spec_write = copy.deepcopy(spec_write)
        self.spec_read = copy.deepcopy(spec_read)

        if dtype is not None:
            self.spec_create["metadata"]["dtype"] = dtype

        if compression is not None:
            if compression == "blosc":
                self.spec_create["metadata"]["compressor"] = {
                    "id": "blosc",
                    "clevel": 3,
                    "cname": "zstd",
                    "shuffle": 2,
                }
        else:
            self.spec_create["metadata"]["compressor"] = None

        if chunks is not None:
            self.spec_create["metadata"]["chunks"] = chunks

        if shape is not None:
            self.spec_create["metadata"]["shape"] = shape

        if "dtype" not in self.spec_create["metadata"]:
            raise ValueError("dtype must be specified")

    def get_path(self, folder):
        return folder / (self.name + ".dat")

    def __get__(self, obj, type=None, version=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + self.name

            if not hasattr(obj, name):
                if version is None:
                    path = self.get_path(obj.path)
                    if path.with_suffix(".indices").exists():
                        version = "csr"
                    else:
                        version = "dense"

                if version == "dense":
                    instance = MemorymapInstance(
                        self.name,
                        self.get_path(obj.path),
                        self.spec_create,
                        self.spec_write,
                        self.spec_read,
                    )
                elif version == "csr":
                    instance = CSRMemorymapInstance(
                        self.name,
                        self.get_path(obj.path),
                        self.spec_create,
                        self.spec_write,
                        self.spec_read,
                    )
                setattr(obj, name, instance)
            return getattr(obj, name)

    def __set__(self, obj, value):
        # if not isinstance(value, np.ndarray):
        #     raise ValueError("Must be an numpy ndarray, not " + str(type(value)))
        path = self.get_path(obj.path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
        if isinstance(value, scipy.sparse._csr.csr_matrix):
            version = "csr"
        else:
            version = "dense"
        self.__get__(obj, version=version)[:] = value

    def exists(self, obj):
        return self.__get__(obj).exists()

    def _repr_html_(self, obj):
        instance = self.__get__(obj)
        return instance._repr_html_()
