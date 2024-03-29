from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .features import FeaturesProcessor

tqdm.pandas()


class Sampler:

    def __init__(self, path_or_df, sample_size=0.1, inference=False):
        self.inference = inference
        self.df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_feather(path_or_df)
        if not self.inference:
            self.df["pct_rank"] = self.df["rank"] / self.df.groupby("id")["cell_id"].transform("count")
        self.presampling(sample_size)
        self.name = ''

    def presampling(self, sample_size):
        nb_ids = self.df.id.unique()
        amount = sample_size if sample_size > 1 else round(len(nb_ids) * sample_size)
        sample_ids = np.random.choice(nb_ids, amount, replace=False)
        self.df = self.df.loc[self.df.id.isin(sample_ids), :]

    def save_dataset(self, dataset):
        today = date.today()
        path = Path(f'data/{today}/')
        path.mkdir(parents=True, exist_ok=True)
        dataset.reset_index().to_feather(path / f'{self.name}')


class PairwiseSampler(Sampler):

    def __init__(self, path, asymmetric=False, sample_method='harmonic', dist_processing='log', filtration='first_md',
                 sample_size=0.1, debug=False, balance_classes=True):

        super().__init__(path, sample_size)

        self.rng = np.random.default_rng()
        self.mms = MinMaxScaler()
        self.asymmetric = asymmetric

        self.classification_bins = (-np.inf, -5, -1, 1, 5, np.inf)
        self.balance = balance_classes & (dist_processing in ['binary', 'multiclass'])

        if balance_classes:
            self.rs = RandomUnderSampler(sampling_strategy='not minority')

        self.mapping = defaultdict(self.def_mapping_value)

        self.mapping.update({
            'log': self.log_dist,
            'binary': self.binary_dist,
            'multiclass': self.multiclass_dist,
            'normalised': self.normalised_dist,
            'harmonic': self.harmonic_sampling,
            'first_md': self.filter_markdown,
            ('first_md', 'second_cd'): self.filter_markdown_code
        })

        self.sampling_method = self.mapping[sample_method]
        self.process_distance = self.mapping[dist_processing]
        self.filter = self.mapping[filtration]

        sym = 'sym' if not asymmetric else 'asym'
        balanced = 'bal' if balance_classes else 'non_bal'
        self.name = f'{dist_processing}_{sym}_{filtration}_{balanced}_pairs.fth'

        self.debug = debug

    def sample_notebook(self, nb_id=None):
        rand_id = nb_id if nb_id else self.df.id.sample(1).values[0]
        nb_df = self.df.loc[self.df.id == rand_id, :]
        return nb_df

    @staticmethod
    def def_mapping_value():
        print('empty or non-existing preprocessing was entered')

        def placeholder(sample, *args, **kwargs):
            return sample

        return placeholder

    @staticmethod
    def filter_markdown(full_sample, nb_df):
        md_ids = nb_df.loc[nb_df.cell_type == 'markdown', 'rank']
        return full_sample[np.isin(full_sample[:, 0], md_ids.values)]

    @staticmethod
    def filter_markdown_code(full_sample, nb_df):
        md_ids = nb_df.loc[nb_df.cell_type == 'markdown', 'rank']
        cd_ids = nb_df.loc[nb_df.cell_type == 'code', 'rank']
        return full_sample[(np.isin(full_sample[:, 0], md_ids.values)) & (np.isin(full_sample[:, 1], cd_ids.values))]

    @staticmethod
    def build_pairwise_distance(nb_df):
        rank_matrix = np.array(np.meshgrid(nb_df['rank'].values, nb_df['rank'].values)).T.reshape(-1, 2)
        rank_matrix = np.append(rank_matrix, (rank_matrix[:, 1] - rank_matrix[:, 0]).reshape(-1, 1), axis=1)
        # pairwise_dist = pd.DataFrame(rank_matrix, columns=['p1', 'p2', 'dist'])
        return rank_matrix

    def harmonic_sampling(self, pairwise_dist):
        sample = []

        p_min = pairwise_dist[:, 2].min()
        p_max = pairwise_dist[:, 2].max()

        dists = range(0, p_max + 1) if self.asymmetric else range(p_min, p_max + 1)
        for rank in dists:
            if rank == 0:
                continue
            amount = np.ceil(p_max / abs(rank))
            subsample = pairwise_dist[pairwise_dist[:, 2] == rank]
            if amount < len(subsample):
                subsample = self.rng.choice(subsample, int(amount))
            sample.append(subsample)

        all_pairs = np.concatenate(sample) if len(sample) > 0 else np.empty((0, 3))
        return all_pairs

    @staticmethod
    def log_dist(distance):
        return np.sign(distance) * np.log(abs(distance) + 0.3)

    def normalised_dist(self, distance):
        return self.mms.fit_transform(distance.reshape(-1, 1))

    @staticmethod
    def binary_dist(distance):
        return np.where((distance == 1) | (distance == -1), 1, 0)

    def multiclass_dist(self, distance):
        return np.digitize(distance, self.classification_bins)

    def balance_classes(self, sample):
        sample, _ = self.rs.fit_resample(sample, sample[:, 2].astype(int).reshape(-1, 1))
        return sample

    def sample_pairs(self, nb_df):
        full_sample = self.build_pairwise_distance(nb_df)
        filtered_sample = self.filter(full_sample, nb_df)
        sample = self.sampling_method(filtered_sample)

        if len(sample) > 0:
            code = nb_df.source.to_numpy()[sample[:, :2]]
            processed_distance = self.process_distance(sample[:, 2])
            calc_sample = np.hstack([code, processed_distance.reshape(-1, 1)])

            if self.balance and (len(np.unique(calc_sample[:, 2])) > 1):
                calc_sample = self.balance_classes(calc_sample)

            return calc_sample

    def sample(self, save=True):
        pairs_df = self.df.groupby('id').progress_apply(self.sample_pairs).explode()
        pairs_df = pairs_df.dropna()
        pairs_df = pd.DataFrame(pairs_df.to_list(), columns=['p1', 'p2', 'score'], index=pairs_df.index)

        if save:
            self.save_dataset(pairs_df.reset_index())

        return pairs_df


class MDSampler(Sampler):
    def __init__(self, path_or_df, sample_size=0.1, inference=False):
        super().__init__(path_or_df, sample_size, inference=inference)

    # @staticmethod
    # def get_code_count(sub_df):
    #     total_code = sub_df[sub_df.cell_type == "code"].shape[0]
    #     return total_code
    #
    # @staticmethod
    # def get_md_count(sub_df):
    #     total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
    #     return total_md
    #
    # @staticmethod
    # def random_global_sample(sub_df, n=20):
    #     code_df = sub_df[sub_df.cell_type == "code"]
    #
    #     n = 20
    #     n = len(code_df) if n > len(code_df) else n
    #     return ' '.join(code_df.source.sample(n).astype(str).tolist())

    @staticmethod
    def calculate_features(grouped_df, feature_list, processor):
        return processor.process(grouped_df, feature_list)

    def sample_ranks(
            self,
            feature_list=('md_count', 'code_count', 'defined_functions',
                          'normalized_plot_functions', 'normalized_defined_functions',
                          'normalized_sloc', 'code_subsample'),
            save=True
    ):
        base_features = ['id', 'cell_id', 'source', 'pct_rank', 'ancestor_id']
        base_features.extend(feature_list)

        processor = FeaturesProcessor()
        feature_df = self.df.groupby('id').apply(self.calculate_features,
                                                 feature_list=feature_list,
                                                 processor=processor)

        markdowns_subset = self.df.merge(feature_df, left_on='id', right_index=True)
        markdowns_subset = markdowns_subset.loc[markdowns_subset.cell_type == 'markdown', :]

        # if save:
        #     self.save_dataset(markdowns_subset.reset_index())

        return markdowns_subset
