import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datasets import Dataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, AutoTokenizer, RobertaTokenizerFast

from data_managment.samplers import MDSampler


class MarkdownDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_path: str = None, val_path: str = None, test_path: str = None, 
        batch_size: int = 32, 
        resample=False,         
        train_dat=None, val_dat=None, test_dat=None, 
        model="distilbert-base-uncased", sample_size=100, val_size=0.1
    ):
        super().__init__()

        self.test_path = test_path
        self.val_path = val_path
        self.train_path = train_path
        self.resample = resample

        self.batch_size = batch_size
        self.validation_size = val_size
        self.padding = 128
        self.sample_size = sample_size

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model, do_lower_case=True)

        self.train_dataset, self.val_dataset, self.test_dataset = train_dat, val_dat, test_dat

    def presampling(self, df, ):
        nb_ids = df.id.unique()
        amount = self.sample_size if self.sample_size > 1 else round(len(nb_ids) * self.sample_size)
        sample_ids = np.random.choice(nb_ids, amount)
        df = df.loc[df.id.isin(sample_ids), :]
        return df

    def _read_train_dataset(self):

        df = pd.read_feather(self.train_path)
            
        df = self.presampling(df)

        if self.resample:
            sampler = MDSampler(df, sample_size=1)
            df = sampler.sample_ranks(save=False)

        df = df.rename(columns={'pct_rank': 'score'})
        
        train_df, val_df = self._split_if_ancestors(df)
        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(val_df)
        return train_dataset, validation_dataset

    def _read_val_dataset(self):

        df = pd.read_feather(self.val_path)
        sampler = MDSampler(df, sample_size=1, inference=False)
        df = sampler.sample_ranks(save=False)
        df = df.rename(columns={'pct_rank': 'score'})
        val_dataset = Dataset.from_pandas(df)
        return val_dataset

    def _read_test_dataset(self):

        df = pd.read_feather(self.test_path)
        sampler = MDSampler(df, sample_size=1, inference=True)
        df = sampler.sample_ranks(save=False)
        test_dataset = Dataset.from_pandas(df)
        return test_dataset

    def tokenize_subsample(self, dataset):

        def flatten_list(hlist):
            return [l for sublist in hlist for l in sublist[1:]]

#         def padding_to_max(ids, max_len=20 * 22):
        def padding_to_max(ids, max_len=20 * 19):
            to_pad = max_len - len(ids)
            ids.extend(to_pad * [0])
            return ids

        def concat_features(example):
            example['input_ids'] = example['input_ids'] + example['ftr_input_ids']
            example['attention_mask'] = example['attention_mask'] + example['ftr_attention_mask']
            return example

        df = dataset.to_pandas()
        mapping = df[['id', 'code_subsample']].drop_duplicates()
        code_subsample = mapping['code_subsample'].tolist()

        code_subsample = [subsample.split('<lop>') for subsample in code_subsample]

        tokenized_code_subsample_ids = []
        tokenized_code_subsample_masks = []

        for subsample in tqdm(code_subsample):
            tokenized = self.tokenizer(subsample,
                                       add_special_tokens=True,
                                       max_length=20,
                                       padding=False,
                                       truncation=True)

            tokenized['input_ids'] = flatten_list(tokenized['input_ids'])
            tokenized['attention_mask'] = flatten_list(tokenized['attention_mask'])

            tokenized['input_ids'] = padding_to_max(tokenized['input_ids'])
            tokenized['attention_mask'] = padding_to_max(tokenized['attention_mask'])

            tokenized_code_subsample_ids.append(tokenized['input_ids'])
            tokenized_code_subsample_masks.append(tokenized['attention_mask'])

        mapping['ftr_input_ids'] = tokenized_code_subsample_ids
        mapping['ftr_attention_mask'] = tokenized_code_subsample_masks

        df = df.merge(mapping[['id', 'ftr_input_ids', 'ftr_attention_mask']], on='id', how='left')
        
        if '__index_level_0__' in df.columns:
            df = df.drop(columns=['__index_level_0__'])

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(concat_features)

        return dataset

    def _preprocess_dataset(self, dataset, inference=False):

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

        dataset = self.tokenize_subsample(dataset)

        cols_to_keep = [
            'input_ids', 'attention_mask', 'md_count', 'code_count',
            'normalized_plot_functions', 'normalized_defined_functions',
            'normalized_sloc'
        ]
        if not inference:
            cols_to_keep += ['score']

        dataset.set_format('pt', cols_to_keep, output_all_columns=True)
        dataset = dataset.rename_column('id', 'notebook_id')
        cols_to_keep.append('notebook_id')
        cols_to_keep.append('cell_id')
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in cols_to_keep])
        return dataset

    def _split_if_ancestors(self, df):
        if 'ancestor_id' in df.columns:
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_size, random_state=0)
            train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
            train_df, val_df = df.iloc[train_ind].reset_index(drop=True), df.iloc[val_ind].reset_index(drop=True)
        else:
            train_df, val_df = train_test_split(df, test_size=0.1)

        return train_df, val_df

    def prepare_data(self):

        if self.train_dataset or self.val_dataset or self.test_dataset:
            return

        if self.train_path and not self.val_path:
            train, val = self._read_train_dataset()
            print('preparing train data')
            self.train_dataset = self._preprocess_dataset(train)
            print('preparing validation data')
            self.val_dataset = self._preprocess_dataset(val)

        if self.val_path:
            val = self._read_val_dataset()
            print("preparing validation data")
            self.val_dataset = self._preprocess_dataset(val)
            
        if self.test_path:
            test = self._read_test_dataset()
            print('preparing test data')
            self.test_dataset = self._preprocess_dataset(test, inference=True)

    # def setup(self, stage=None):
    #     if stage == 'fit' or stage is None:
    #         pass
    #
    #     if stage == 'test' or stage is None:
    #         pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, num_workers=4,
                          pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, num_workers=4,
                          pin_memory=False)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )
# %%
