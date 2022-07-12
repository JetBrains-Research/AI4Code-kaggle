import pandas as pd
import pytorch_lightning as pl
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from data_managment.samplers import MDSampler


class MarkdownDataModule(pl.LightningDataModule):
    def __init__(self, train_path: str = None, test_path: str = None, batch_size: int = 32, resample=False,
                 train_dat=None, val_dat=None, test_dat=None, model="distilbert-base-uncased", sample_size=100):
        super().__init__()

        self.test_path = test_path
        self.train_path = train_path
        self.resample = resample

        self.batch_size = batch_size
        self.validation_size = 0.1
        self.padding = 128
        self.sample_size = sample_size

        self.tokenizer = DistilBertTokenizer.from_pretrained(model, do_lower_case=True)

        self.train_dataset, self.val_dataset, self.test_dataset = train_dat, val_dat, test_dat

    def _read_train_dataset(self):

        df = pd.read_feather(self.train_path)
        df = df.sample(self.sample_size)

        if self.resample:
            sampler = MDSampler(df, sample_size=1)
            df = sampler.sample_ranks(save=False)

        df = df.rename(columns={'pct_rank': 'score'})
        train_df, val_df = self._split_if_ancestors(df)

        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(val_df)

        return train_dataset, validation_dataset

    def _read_test_dataset(self):

        df = pd.read_feather(self.test_path)
        sampler = MDSampler(df, sample_size=1)
        df = sampler.sample_ranks(save=False)
        df = df.rename(columns={'pct_rank': 'score'})
        test_dataset = Dataset.from_pandas(df)
        return test_dataset

    def _preprocess_dataset(self, dataset):

        def process_batch(batch):
            tokenized = self.tokenizer(
                batch['source'],
                padding='max_length',
                truncation=True,
                max_length=self.padding
            )
            return tokenized

        dataset = dataset.map(
            lambda batch: process_batch(batch),
            batched=True, batch_size=self.batch_size,
        )

        dataset.set_format('pt', ['input_ids', 'attention_mask', 'md_count', 'code_count', 'defined_functions',
                                  'normalized_plot_functions', 'normalized_defined_functions',
                                  'normalized_sloc', 'score'])

        return dataset

    def _split_if_ancestors(self, df):
        #
        # if 'ancestor_id' in df.columns:
        #
        #     splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_size, random_state=0)
        #     train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
        #     train_df, val_df = df.loc[train_ind].reset_index(drop=True), df.loc[val_ind].reset_index(drop=True)
        #
        # else:

        train_df, val_df = train_test_split(df, test_size=0.1)

        return train_df, val_df

    def prepare_data(self):

        if (self.train_dataset is not None) and (self.val_dataset is not None) and (self.test_dataset is not None):
            return
        train, val = self._read_train_dataset()
        test = self._read_test_dataset()
        print('preparing train data')
        self.train_dataset = self._preprocess_dataset(train)
        print('preparing validation data')
        self.val_dataset = self._preprocess_dataset(val)
        print('preparing test data')
        self.test_dataset = self._preprocess_dataset(test)

    # def setup(self, stage=None):
    #     if stage == 'fit' or stage is None:
    #         pass
    #
    #     if stage == 'test' or stage is None:
    #         pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, num_workers=4,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, num_workers=4,
                          pin_memory=True)

# %%
