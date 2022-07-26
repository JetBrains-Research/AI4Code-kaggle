import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from models.transformers.auto_listwise_model import ListwiseModel, CleanListwisePredictionCallback
from models.transformers.listwise_dataset import NotebookDataset, collate_fn
from prepare_listwise_dataset import prepare_listwise_dataset
import argparse
from omegaconf import OmegaConf

from prepare_listwise_dataset_constants import BATCH_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("device", type=int)
args = parser.parse_args()
config = OmegaConf.load(args.config)

print(config)

model_name = config.get('model', 'distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading train dataset")
train_dataset = NotebookDataset(
    prepare_listwise_dataset(model_name, "data/all_dataset/train_df.fth"),
    tokenizer.pad_token_id
)
print("Loading val dataset")
val_dataset = NotebookDataset(
    prepare_listwise_dataset(model_name, "data/all_dataset/val_df.fth"),
    tokenizer.pad_token_id,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    pin_memory=False,
    shuffle=True,
    num_workers=1,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
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
        model_name=model_name,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
else:
    model = ListwiseModel(
        model_name=model_name,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    
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

trainer.fit(model, train_loader, val_loader)
