import pandas as pd
import yaml
import sys, os
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset

BERT_PATH = "../input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased"
NVALID = 0.1  # size of validation set
MAX_LEN = 128


class MarkdownModel(nn.Module):
    def __init__(self):
        super(MarkdownModel, self).__init__()
        self.distill_bert = DistilBertModel.from_pretrained(BERT_PATH)
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.distill_bert(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x


class MarkdownDataset(Dataset):

    def __init__(self, df, max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 5:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr


def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999),
                                 eps=1e-08)
    return optimizer


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)

    optimizer = get_optimizer(model)

    criterion = torch.nn.MSELoss()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)

        lr = adjust_lr(optimizer, e)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            optimizer.zero_grad()
            pred = model(inputs[0], inputs[1])

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {lr}")

        y_val, y_pred = validate(model, val_loader)

        print("Validation MSE:", np.round(mean_squared_error(y_val, y_pred), 4))
        print()
    return model, y_pred


def run_baseline(train_path):
    df = pd.read_feather(train_path)

    splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
    train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
    train_df, val_df = df.loc[train_ind].reset_index(drop=True), df.loc[val_ind].reset_index(drop=True)

    train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
    val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

    train_ds = MarkdownDataset(train_df_mark, max_len=MAX_LEN)
    val_ds = MarkdownDataset(val_df_mark, max_len=MAX_LEN)

    BS = 32
    NW = 1

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=NW,
                            pin_memory=False, drop_last=False)

    model = MarkdownModel()
    model = model.cuda()
    model, y_pred = train(model, train_loader, val_loader, epochs=1)

    return model, y_pred


if __name__ == "__main__":
    with open("paths.yaml", "r") as stream:
        try: path = yaml.safe_load(stream)['train_path']
        except yaml.YAMLError as exc: print(exc)

    model, y_pred = run_baseline(path)
