from pathlib import Path
import pytorch_lightning as pl
from models.kaggle_rank_baseline.rank_datamodule import MarkdownDataModule
from models.transformers.auto_ranking_model import AutoRankingModel
from datasets import Dataset
import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()
config = OmegaConf.load(args.config)

cols_to_keep =  [
    'input_ids', 'attention_mask', 'md_count', 'code_count',
    'normalized_plot_functions', 'normalized_defined_functions',
    'normalized_sloc', 'score'
]
train_dat = Dataset.load_from_disk("data/full_dataset/full_train_rank_dataset.dat")
train_dat.set_format('pt', cols_to_keep, output_all_columns=True)
val_dat = Dataset.load_from_disk("data/full_dataset/full_val_rank_dataset.dat")
val_dat.set_format('pt', cols_to_keep, output_all_columns=True)

data_module = MarkdownDataModule(
    train_dat = train_dat,
    val_dat = val_dat,
    batch_size=config.get('batch_size', 32),
)

scheduler_config = config.get('scheduler_config')
dropout_rate = config.get('dropout_rate', 0.)

model = AutoRankingModel(
    optimizer_config=config.optimizer_config,
    scheduler_config=scheduler_config,
    dropout_rate=0.
)

config_filename = args.config.split('/')[-1][:-4]
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f'checkpoints/{config_filename}/',
    filename='{epoch}-{step}-{val_kendall_tau}',
    save_top_k=-1,
    save_on_train_epoch_end=False,
)

wandb_logger = pl.loggers.WandbLogger(project="JupyterBert", entity="jbr_jupyter")
# wandb_logger.experiment.config.update(config.optimizer_config)
# wandb_logger.experiment.config.update(config.scheduler_config)

trainer = pl.Trainer(
    logger=wandb_logger, 
    accelerator="gpu",
    max_epochs=10,
    devices=[config.device],
    enable_progress_bar=True,
    log_every_n_steps=1,
    val_check_interval=10000,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
)

trainer.fit(model, data_module)
