import json

import pandas as pd
from pandas.testing import assert_series_equal
from tqdm import tqdm


class Dataset:
    def __init__(
        self, read_path, clean_forks=False, inference=False, n=None, notebook_ids=None
    ):

        self.test = inference
        self.n = n
        self.notebook_ids = notebook_ids
        self.data = self._read_subset(read_path)

        if not self.test:
            self.order_df = pd.read_csv(
                read_path.parents[0] / "train_orders.csv", index_col="id"
            )
            self.order_df = self.order_df.cell_order.apply(str.split)

            self.ancestors = pd.read_csv(read_path.parents[0] / "train_ancestors.csv")
            self.clean_forks = clean_forks

    @staticmethod
    def __read_notebook(path):
        with open(path) as file:
            data = json.load(file)
            df = pd.DataFrame.from_dict(data)
            df = df.reset_index().rename(columns={"index": "cell_id"})
            df = df.assign(id=path.stem)

            return df

    def _read_subset(self, data_path):
        if self.notebook_ids is None:
            paths_train = list(data_path.glob("*.json"))
            paths_train = paths_train[: self.n] if self.n else paths_train
        else:
            paths_train = [
                data_path / f"{notebook_id}.json" for notebook_id in self.notebook_ids
            ]

        dfs = [self.__read_notebook(path) for path in tqdm(paths_train)]
        main_df = pd.concat(dfs)
        # if save:
        #     main_df.to_csv(f'{dataset}_dataset.csv')

        return main_df

    def _place_forks(self):
        self.data = self.data.merge(self.ancestors, on="id")

    def _delete_forks(self):
        firsts = self.ancestors.groupby("ancestor_id").apply(lambda x: x.id.iloc[0])
        self.data = self.data.loc[self.data.id.isin(firsts), :]

    def _rank_cells(self):
        self.data["rank"] = self.data.groupby("id").cumcount()

    def _save_as_feather(self, path):
        name = "test_dataset.fth" if self.test else "train_dataset.fth"
        self.data.reset_index().to_feather(path / name)

    def _order_notebooks(self, check_order=False):

        order = (
            self.order_df.explode()
            .reset_index()
            .rename(columns={"cell_order": "cell_id"})
        )
        self.data = order.merge(self.data, on=["id", "cell_id"])

        if check_order:
            try:
                assert_series_equal(self.data.cell_id, order.cell_id)
            except AssertionError:
                print("order is incorrect")

    def prepare_and_save_data(self, save_path=None):

        if not self.test:
            self._order_notebooks()
            self._rank_cells()
            self._place_forks()

            if self.clean_forks:
                self._delete_forks()

        if save_path:
            self._save_as_feather(save_path)

        return self.data
