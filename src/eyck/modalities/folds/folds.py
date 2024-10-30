import numpy as np
import pandas as pd

from eyck.flow import (
    Flow,
    Stored,
)


class Folds(Flow):
    """
    Folds of multiple cell and reion combinations
    """

    folds: dict = Stored()
    """The folds"""

    def sample_cells(
        self,
        obs: pd.DataFrame,
        n_folds: int,
        n_repeats: int = 1,
        overwrite: bool = False,
        seed: int = 1,
    ):
        """
        Sample cells and regions into folds

        Parameters:
            obs:
                the obs
            n_folds:
                the number of folds
            n_repeats:
                the number of repeats
            overwrite:
                whether to overwrite existing folds
        """
        if not overwrite and self.get("folds").exists(self):
            return self

        folds = []

        for repeat_ix in range(n_repeats):
            generator = np.random.RandomState(repeat_ix * seed)

            cells_all = generator.permutation(obs.n_cells)

            cell_bins = np.floor(
                (np.arange(len(cells_all)) / (len(cells_all) / n_folds))
            )

            for i in range(n_folds):
                cells_train = cells_all[cell_bins != i]
                cells_validation_test = cells_all[cell_bins == i]
                cells_validation = cells_validation_test[
                    : (len(cells_validation_test) // 2)
                ]
                cells_test = cells_validation_test[(len(cells_validation_test) // 2) :]

                folds.append(
                    {
                        "cells_train": cells_train,
                        "cells_validation": cells_validation,
                        "cells_test": cells_test,
                        "repeat": repeat_ix,
                    }
                )
        self.folds = folds

        return self

    def __getitem__(self, ix):
        return self.folds[ix]

    def __len__(self):
        return len(self.folds)
