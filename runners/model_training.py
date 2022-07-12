import pandas as pd
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from data_managment.preprocessing import kaggle_cleaning
from .model_evaluation import PairwiseKendallTauEvaluator


class PairwiseTrainingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        return self.df["tokenized"].iloc[ind], self.df["score"].iloc[ind]

    @staticmethod
    def get_collate_fn(tokenizer):
        def collate_fn(batch):
            tokens = [{"input_ids": x[0]} for x in batch]
            scores = torch.LongTensor([x[1] for x in batch]).view(-1, 1)
            res = tokenizer.pad(tokens, return_attention_mask=True, return_tensors="pt")
            return {
                "score": scores,
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"],
            }

        return collate_fn


class PairwiseTrainingDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_datasets, batch_size, collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            for val_dataset in self.val_datasets
        ]


class PairwiseKendallTauTrainer:
    def __init__(
        self,
        path_to_train_df,
        paths_to_val_df,
        tokenizer,
        model,
        max_p_length,
        batch_size,
    ):
        self.df = self._load_df(path_to_train_df, tokenizer, max_p_length)
        self.val_dfs = [
            self._load_df(path, tokenizer, max_p_length) for path in paths_to_val_df
        ]

        self.training_dataset = PairwiseTrainingDataset(self.df)
        self.val_datasets = [PairwiseTrainingDataset(val_df) for val_df in self.val_dfs]
        self.data_module = PairwiseTrainingDataModule(
            self.training_dataset,
            self.val_datasets,
            batch_size,
            PairwiseTrainingDataset.get_collate_fn(tokenizer),
        )

        self.model = model

    def train(self, **trainer_config):
        wandb_logger = WandbLogger(project="JupyterBert", entity="jbr_jupyter")
        trainer = pl.Trainer(logger=wandb_logger, **trainer_config)
        trainer.fit(self.model, self.data_module)

    def _load_df(self, path_to_df, tokenizer, max_p_length):
        df = pd.read_feather(path_to_df)
        df["p1"] = df["p1"].apply(lambda s: kaggle_cleaning(s))
        df["p2"] = df["p2"].apply(lambda s: kaggle_cleaning(s))
        df["p1_tokenized"] = df["p1"].apply(
            lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][:max_p_length]
        )
        df["p2_tokenized"] = df["p2"].apply(
            lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][:max_p_length]
        )
        df["tokenized"] = df.apply(
            lambda row: [tokenizer.cls_token_id]
            + row["p1_tokenized"]
            + [tokenizer.sep_token_id]
            + row["p2_tokenized"]
            + [tokenizer.sep_token_id],
            axis=1,
        )
        return df
