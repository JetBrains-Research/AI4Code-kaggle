from collections import OrderedDict

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from models.transformers.auto_listwise_model import ListwiseModel, CleanListwisePredictionCallback
from models.transformers.listwise_dataset import NotebookDataset, collate_fn
from prepare_listwise_dataset import prepare_listwise_dataset
import argparse
from omegaconf import OmegaConf
import pickle
import numpy as np
from dataclasses import dataclass

from prepare_listwise_dataset_constants import BATCH_SIZE


@dataclass
class ProcessedNotebook:
    code_tokens: list
    code_scores: list
    md_tokens: torch.tensor
    md_scores: torch.tensor

    md_cell_ids: np.ndarray

    n_md: int
    n_code: int
    notebook_id: str = ""
        
class NotebookDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        datapoints,
        sep_token_id,
        pad_token_id,
        md_len,
        code_len,
        total_md_len,
        total_code_len,
    ):
        self.datapoints = datapoints
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        
        self.md_len = md_len
        self.code_len = code_len
        
        self.total_md_len = total_md_len
        self.total_code_len = total_code_len
        self.total_len = total_md_len + total_code_len

        self.select_md = self.total_md_len // self.md_len
        self.select_code = self.total_code_len // self.code_len

        self.reps = self.select_md

        self.n_examples = 0
        for i, datapoint in tqdm(enumerate(datapoints)):
            
            n_md = len(datapoint.md_tokens)
            n_code = len(datapoint.code_tokens)
            
            self.n_examples += n_md

        self.notebook_indices = torch.zeros(self.n_examples, dtype=torch.int)
        cur_len = 0
        for i, datapoint in enumerate(datapoints):
            n_md = datapoint.md_tokens.size(0)
            self.notebook_indices[cur_len:cur_len + n_md] = i
            cur_len += n_md

        self.selected_permutations = torch.zeros(self.n_examples, self.select_md, dtype=torch.long)
        self.reset_dataset()

    def __len__(self):
        return self.n_examples

    def reset_dataset(self):
        cur_len = 0
        for i, datapoint in enumerate(self.datapoints):
            n_md = datapoint.md_tokens.size(0)
            self.selected_permutations[cur_len:cur_len + n_md, :] = torch.cat([
                torch.randperm(n_md) for _ in range(self.reps)
            ]).view(-1, self.select_md)
            cur_len += n_md

    @staticmethod
    def select_n(tokens, scores, max_len, keep_order):
        n_tokens = tokens.size(0)
        len_tokens = tokens.size(1)

        n_selected = max_len // len_tokens

        if n_selected >= n_tokens:
            if keep_order:
                return tokens, scores
            else:
                indices = torch.randperm(n_tokens)
                return tokens[indices], scores[indices]

        if keep_order:
            middle_inds = np.random.choice(n_tokens - 2, n_selected - 2, replace=False)
            middle_inds.sort()
            indices = torch.cat((
                torch.tensor([0]),
                torch.tensor(middle_inds + 1),
                torch.tensor([n_tokens - 1])
            ))
            return tokens[indices], scores[indices]
        else:
            indices = torch.randperm(n_tokens)[:n_selected]
            return tokens[indices], scores[indices]

    def __getitem__(self, ind):
        notebook_ind = self.notebook_indices[ind]
        datapoint = self.datapoints[notebook_ind]
        permutation = self.selected_permutations[ind]

        code_tokens, code_scores = self.select_n(
            datapoint.code_tokens, datapoint.code_scores, self.total_code_len, True
        )

        md_tokens = datapoint.md_tokens[permutation]
        md_scores = datapoint.md_scores[permutation]
        md_cell_ids = datapoint.md_cell_ids[permutation]

        input_ids = torch.full((self.total_len,), self.pad_token_id)
        input_ids[:md_tokens.numel()] = md_tokens.view(-1)
        input_ids[self.total_md_len:self.total_md_len + code_tokens.numel()] = code_tokens.view(-1)

        attention_mask = (input_ids != self.pad_token_id).type(torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': md_scores,
            'cell_ids': md_cell_ids,
            'notebook_id': datapoint.notebook_id,
        }


        
parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("device", type=int)
args = parser.parse_args()
config = OmegaConf.load(args.config)

print(config)

model_name = config.get('model', 'distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(model_name)

md_total = config.get("md_total", 256)
md_len = config.get("md_len", 32)
code_len = config.get("code_len", 32)

code_ids = pickle.load(open("data/300k_dataset/code_ids.pkl", "rb"))

print("Loading train dataset")
train_dataset = pickle.load(open(f"data/300k_dataset/train_{md_len}_{code_len}_graphcodebert.pkl", "rb"))
# train_dataset = NotebookDataset(
#     prepare_listwise_dataset(model_name, "data/all_dataset/train_df.fth"),
#     tokenizer.pad_token_id
# )
print("Loading val dataset")
val_dataset = pickle.load(open(f"data/300k_dataset/val_{md_len}_{code_len}_graphcodebert.pkl", "rb"))
# val_dataset = NotebookDataset(
#     prepare_listwise_dataset(model_name, "data/all_dataset/val_df.fth"),
#     tokenizer.pad_token_id,
# )

print("Computing code sizes")
n_code = {}
for dp in train_dataset.datapoints:
    n_code[dp.notebook_id] = len(dp.code_tokens)
for dp in val_dataset.datapoints:
    n_code[dp.notebook_id] = len(dp.code_tokens)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.get('batch_size', 32),
    collate_fn=collate_fn,
    pin_memory=False,
    shuffle=True,
    num_workers=1,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.get('batch_size', 32),
    collate_fn=collate_fn,
    pin_memory=False,
    shuffle=False,
    num_workers=1,
)

optimizer_config = config.get('optimizer_config')
# scheduler_config = config.get('scheduler_config')
training_steps = config.get(
    "training_steps",
    config.get("max_epochs", 1) * len(train_loader)
)
scheduler_config = {
    "warmup_steps": 0.05 * training_steps,
    "training_steps": training_steps,
    "cur_step": config.get("cur_step", 0)
}

dropout_rate = config.get('dropout_rate', 0.)
use_features = config.get("use_features", False)

ckpt = config.get("checkpoint")
if ckpt:
    print(f"Loading from {ckpt}")
    model = ListwiseModel.load_from_checkpoint(
        ckpt,
        md_len=md_len,
        md_total=md_total,
        model_name=model_name,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        code_ids=code_ids,
    )
else:
    model = ListwiseModel(
        md_len=md_len,
        md_total=md_total,
        model_name=model_name,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        code_ids=code_ids,
    )


def fix_state(state_dict):
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("distill_bert"):
            new_state[k.replace("distill_bert", "model")] = v
        elif k.startswith("dense"):
            new_state[k.replace("dense", "linear")] = v
        else:
            new_state[k] = v
#             raise ValueError(f"Unexpected key: {k}")
    return new_state


automodel_checkpoint = config.get("automodel_checkpoint")
if automodel_checkpoint:
    fixed_state = fix_state(torch.load(automodel_checkpoint)["state_dict"])
    model.load_state_dict(fixed_state)


config_filename = args.config[7:-4].replace("/", "_")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f'checkpoints/{config_filename}/',
    filename='{epoch:02d}-{step}-{val_kendall_tau:.5f}',
    save_top_k=-1,
    save_on_train_epoch_end=False,
)
checkpoint_train_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f'checkpoints/{config_filename}/',
    filename='{epoch:02d}-{step}-{val_kendall_tau:.5f}',
    save_top_k=-1,
    save_on_train_epoch_end=True,
)

# wandb_logger = pl.loggers.WandbLogger(project="JupyterBert", entity="jbr_jupyter")
wandb_logger = pl.loggers.WandbLogger(project="JupyterBert")

trainer = pl.Trainer(
    logger=wandb_logger, 
    accelerator="gpu",
    max_epochs=config.get("max_epochs", 1),
    devices=[args.device],
    enable_progress_bar=True,
    log_every_n_steps=20,
    val_check_interval=config.get("val_check_interval", 10000),
    callbacks=[checkpoint_callback, checkpoint_train_callback, CleanListwisePredictionCallback()],
    accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
    # resume_from_checkpoint=ckpt,
)
# trainer.validate(model, val_loader)
trainer.fit(model, train_loader, val_loader)
