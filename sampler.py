import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

tqdm.pandas()


class Sampler:

    def __init__(self, path, asymmetric=False, sample_method='harmonic', dist_processing='log', filter='markdown',
                 sample_size=0.1):
        self.df = pd.read_feather(path)
        self.presampling(sample_size)
        self.rng = np.random.default_rng()
        self.mms = MinMaxScaler()
        self.name = 'data'
        self.asymmetric = asymmetric

        self.mapping = {
            'log': self.log_dist,
            'normalised': self.normalised_dist,
            'harmonic': self.harmonic_sampling_new,
            'markdown': self.filter_markdown
        }

        self.sampling_method = self.mapping[sample_method]
        self.process_distance = self.mapping[dist_processing]
        self.filter = self.mapping[filter]

    def presampling(self, sample_size):
        nb_ids = self.df.id.unique()
        amount = sample_size if sample_size > 1 else round(len(nb_ids) * sample_size)
        sample_ids = np.random.choice(nb_ids, amount)
        self.df = self.df.loc[self.df.id.isin(sample_ids), :]

    @staticmethod
    def filter_markdown(nb_df, full_sample):
        md_ids = nb_df.loc[nb_df.cell_type == 'markdown', 'rank']
        return full_sample[np.isin(full_sample[:, 0], md_ids.values)]

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

        all_pairs = np.concatenate(sample)
        return all_pairs

    @staticmethod
    def log_dist(distance):
        return np.sign(distance) * np.log(abs(distance) + 0.3)

    def normalised_dist(self, distance):
        return self.mms.fit_transform(distance.reshape(-1, 1))

    def sample_pairs(self, nb_df):
        # print(nb_df.id)
        full_sample = self.build_pairwise_distance(nb_df)
        filtered_sample = self.filter(nb_df, full_sample)
        sample = self.sampling_method(filtered_sample)
        if len(sample) > 0:
            code = nb_df.source.to_numpy()[sample[:, :2]]
            distance = self.process_distance(sample[:, 2])
            sample = np.hstack([code, distance.reshape(-1, 1)])
            return sample

    def sample(self, save=True):
        pairs_df = self.df.groupby('id').progress_apply(self.sample_pairs).explode()
        pairs_df = pairs_df.dropna()
        pairs_df = pd.DataFrame(pairs_df.to_list(), columns=['p1', 'p2', 'score'], index=pairs_df.index)
        #
        # if save:
        #     self.save_dataset()

        return pairs_df

    def name_generator(self):
        pass

    def save_dataset(self):
        pass
