import json

import pandas as pd
from pandas.testing import assert_series_equal
from tqdm import tqdm


class Dataset:

    def __init__(self, read_path, save_path, clean_forks=False):

        self.data = self._read_all_data(read_path)

        self.order_df = pd.read_csv(read_path / 'train_orders.csv', index_col='id')
        self.order_df = self.order_df.cell_order.apply(str.split)

        self.ancestors = pd.read_csv(read_path / 'train_ancestors.csv')

        self.save_path = save_path
        self.clean_forks = clean_forks

    @staticmethod
    def __read_notebook(path):
        with open(path) as file:
            data = json.load(file)
            df = pd.DataFrame.from_dict(data)
            df = df.reset_index().rename(columns={'index': 'cell_id'})
            df = df.assign(id=path.stem)

            return df

    def _read_subset(self, data_path, dataset, save=True):
        paths_train = list((data_path / dataset).glob('*.json'))
        dfs = [self.__read_notebook(path) for path in tqdm(paths_train)]
        main_df = pd.concat(dfs)
        if save:
            main_df.to_csv(f'{dataset}_dataset.csv')

        return main_df

    def _read_all_data(self, path):
        train = self._read_subset(path, 'train')
        test = self._read_subset(path, 'test')

        return {'train': train, 'test': test}

    def _place_forks(self):
        self.data['train'] = self.data['train'].merge(self.ancestors, on='id')

    def _delete_forks(self):
        firsts = self.ancestors.groupby('ancestor_id').apply(lambda x: x.id.iloc[0])
        self.data['train'] = self.data['train'].loc[self.data['train'].id.isin(firsts), :]

    def _rank_cells(self):
        self.data['train']['rank'] = self.data['train'].groupby('id').cumcount()
        self.data['test']['rank'] = self.data['test'].groupby('id').cumcount()

    def _save_as_feather(self):
        self.data['train'].reset_index().to_feather(self.save_path / 'train_dataset.fth')
        self.data['test'].reset_index().to_feather(self.save_path / 'test_dataset.fth')

    def _order_notebooks(self, check_order=False):

        order = self.order_df.explode().reset_index().rename(columns={'cell_order': 'cell_id'})
        self.data['train'] = order.merge(self.data['train'], on=['id', 'cell_id'])

        if check_order:
            try:
                assert_series_equal(self.data['train'].cell_id, order.cell_id)
            except AssertionError:
                print('order is incorrect')

    def prepare_and_save_data(self):
        self._order_notebooks()
        self._rank_cells()
        self._place_forks()

        if self.clean_forks:
            self._delete_forks()

        self._save_as_feather()
