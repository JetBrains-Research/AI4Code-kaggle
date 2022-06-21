import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

tqdm.pandas()


class Sampler:

    def __init__(self, path, asymmetric=False, sample_method='harmonic', dist_processing='log', sample_size=0.1):
        self.df = pd.read_feather(path)
        self.presampling(sample_size)
        self.rng = np.random.default_rng()
        self.mms = MinMaxScaler()
        self.name = 'data'
        self.asymmetric = asymmetric

        self.mapping = {
            'log': self.log_dist,
            'normalised': self.normalised_dist,
            'harmonic': self.harmonic_sampling
        }

        self.sampling_method = self.mapping[sample_method]
        self.process_distance = self.mapping[dist_processing]

    def presampling(self, sample_size):
        nb_ids = self.df.id.unique()
        amount = sample_size if sample_size > 1 else round(len(nb_ids) * sample_size)
        sample_ids = np.random.choice(nb_ids, amount)
        self.df = self.df.loc[self.df.id.isin(sample_ids), :]

    def harmonic_sampling(self, rank_matrix):
        all_pairs = []
        for rank in range(1, len(rank_matrix)):
            mask = (rank_matrix == rank) if self.asymmetric else ((rank_matrix == rank) | (rank_matrix == rank * -1))
            pairs = list(zip(*np.nonzero(mask)))
            amount = np.ceil(len(rank_matrix) / rank)
            pairs = self.rng.choice(pairs, int(amount))
            all_pairs.append(pairs)
        all_pairs = np.concatenate(all_pairs)

        return all_pairs

    @staticmethod
    def log_dist(distance):
        return np.sign(distance) * np.log(abs(distance))

    def normalised_dist(self, distance):
        return self.mms.fit_transform(distance.reshape(-1, 1))

    def calc_distance(self, pairs):
        distance = pairs[:, 1] - pairs[:, 0]
        distance = abs(distance) if self.asymmetric else distance
        return distance

    def sample_pairs(self, nb_df):
        rank_matrix = (nb_df['rank'].values - nb_df['rank'].values[:, None])
        pairs = self.sampling_method(rank_matrix)
        cell_pairs = nb_df.source.to_numpy()[pairs]
        distance = self.calc_distance(pairs)
        distance = self.process_distance(distance)
        result = np.c_[cell_pairs, pairs, distance]
        return result

    def sample(self):
        pairs_df = self.df.groupby('id').progress_apply(self.sample_pairs).explode()
        pairs_df = pd.DataFrame(pairs_df.to_list(), columns=['p1', 'p2', 'rank1', 'rank2', 'score'],
                                index=pairs_df.index)
        return pairs_df
