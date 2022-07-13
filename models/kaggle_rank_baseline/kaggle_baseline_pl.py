import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import LightningModule, Trainer
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer

BERT_PATH = "../input/huggingface-bert_classification-variants/distilbert-base-uncased/distilbert-base-uncased"
NVALID = 0.1  # size of validation set
MAX_LEN = 128


class MarkdownModelPl(LightningModule):
    def __init__(self):
        super(MarkdownModelPl, self).__init__()
        self.distill_bert = DistilBertModel.from_pretrained(BERT_PATH)
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x

    def training_step(self, batch, batch_idx):
        inputs, target = self.__read_data(batch)
        pred = self(inputs[0], inputs[1])

        loss = torch.nn.MSELoss(pred, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        return optimizer

    @staticmethod
    def __read_data(data):
        return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


class MarkdownDataModule(pl.LightningDataModule):
    def __init__(self, train_path: str, batch_size: int = 32):
        super().__init__()

        self.train_path = train_path
        self.batch_size = batch_size

        self.train_dataset, self.val_dataset = None, None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._prepare_dataset()

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True
        )

    def _prepare_dataset(self):
        df = pd.read_feather(self.train_path)

        splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
        train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
        train_df, val_df = df.loc[train_ind].reset_index(drop=True), df.loc[
            val_ind
        ].reset_index(drop=True)

        train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(
            drop=True
        )
        val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

        train_ds = MarkdownDataset(train_df_mark, max_len=MAX_LEN)
        val_ds = MarkdownDataset(val_df_mark, max_len=MAX_LEN)

        self.train_dataset, self.val_dataset = train_ds, val_ds


class MarkdownDataset(Dataset):
    def __init__(self, df, max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            BERT_PATH, do_lower_case=True
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = torch.LongTensor(inputs["input_ids"])
        mask = torch.LongTensor(inputs["attention_mask"])

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


def run_baseline(train_path):
    MDM = MarkdownDataModule(train_path)
    model = MarkdownModelPl()

    trainer = Trainer()
    trainer.fit(model, MDM)
    return model, None


if __name__ == "__main__":
    with open("paths.yaml", "r") as stream:
        try:
            path = yaml.safe_load(stream)["train_path"]
        except yaml.YAMLError as exc:
            print(exc)

    model, y_pred = run_baseline(path)
