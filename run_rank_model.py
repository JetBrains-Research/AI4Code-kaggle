from pathlib import Path
import pytorch_lightning as pl
from models.kaggle_rank_baseline.rank_datamodule import MarkdownDataModule
from models.transformers.auto_ranking_model import AutoRankingModel
from datasets import Dataset
import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("device", type=int)
args = parser.parse_args()
config = OmegaConf.load(args.config)

print(config)

model = config.get('model', 'distilbert-base-uncased')

cols_to_keep =  [
    'input_ids', 'attention_mask', 'md_count', 'code_count', 'score'
]

train_dataset_paths = {
    'distilbert-base-uncased': "data/all_dataset/distilbert_train_rank_dataset.dat",
    'microsoft/unixcoder-base': "data/all_dataset/unixcoder_train_rank_dataset.dat",
    "microsoft/codebert-base": "data/all_dataset/codebert_train_rank_dataset.dat",
    "microsoft/graphcodebert-base": "data/all_dataset/graphcodebert_train_rank_dataset.dat",
}
val_dataset_paths = {
    'distilbert-base-uncased': "data/all_dataset/distilbert_val_rank_dataset.dat",
    'microsoft/unixcoder-base': "data/all_dataset/unixcoder_val_rank_dataset.dat",
    "microsoft/codebert-base": "data/all_dataset/codebert_val_rank_dataset.dat",
    # "microsoft/graphcodebert-base": "data/all_dataset/graphcodebert_val_small_rank_dataset.dat",
    "microsoft/graphcodebert-base": "data/all_dataset/graphcodebert_val_rank_dataset.dat",
}

print("Loading train dataset")
train_dat = Dataset.load_from_disk(train_dataset_paths[model])
train_dat.set_format('pt', cols_to_keep, output_all_columns=True)
print("Loading val dataset")
val_dat = Dataset.load_from_disk(val_dataset_paths[model])
val_dat.set_format('pt', cols_to_keep, output_all_columns=True)

print("Creating data module")
data_module = MarkdownDataModule(
    train_dat = train_dat,
    val_dat = val_dat,
    batch_size=config.get('batch_size', 32),
    model=model,
)

optimizer_config = config.get('optimizer_config')
# scheduler_config = config.get('scheduler_config')
training_steps = config.get(
    "training_steps",
    config.get("max_epochs", 1) * len(data_module.train_dataloader())
)
scheduler_config = {
    "warmup_steps": 0.05 * training_steps,
    "training_steps": training_steps,
}

dropout_rate = config.get('dropout_rate', 0.)
use_features = config.get("use_features", False)

ckpt = config.get("checkpoint")
if ckpt:
    print(f"Loading from {ckpt}")
    model = AutoRankingModel.load_from_checkpoint(
        ckpt,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        model=model,
    )
else:
    model = AutoRankingModel(
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        model=model,
    )
    
config_filename = args.config[7:-4].replace("/", "_")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f'checkpoints/{config_filename}/',
    filename='{epoch:02d}-{step}-{val_kendall_tau:.5f}',
    save_top_k=-1,
    save_on_train_epoch_end=False,
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
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
)

trainer.fit(model, data_module)
