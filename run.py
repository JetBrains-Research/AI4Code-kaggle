import wandb
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from datasets import Dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast, AdamW
from torch.utils.data import DataLoader
import torch
from torchmetrics import R2Score
import datasets
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


datasets.logging.disable_progress_bar()

class CodeBERTBaseLine(pl.LightningModule):

    def __init__(self, model="microsoft/codebert-base"):
        super(CodeBERTBaseLine, self).__init__()

        self.codebert = RobertaModel.from_pretrained(model,
                                                     hidden_dropout_prob=0.,
                                                     attention_probs_dropout_prob=0.,
                                                     output_attentions=False,
                                                     output_hidden_states=False,
                                                     return_dict=True)
 

#        self.dropout = torch.nn.Dropout(0.3)
        self.dropout = torch.nn.Dropout(0.0)
        self.dense = torch.nn.Linear(768, 200)
        self.regression = torch.nn.Linear(200, 1)
        # self.full_regression = torch.nn.Linear(768, 1)
        # self.regression2 = torch.nn.Linear(100, 1)
        self.activation = torch.nn.ReLU()
        # self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.L1Loss()
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.score = R2Score()

    def _calculate_major_class(self, x, process_stage):
        major_class_percent = np.around(x.unique(return_counts=True)[1].max().cpu(), 5) / len(x)
        self.log(f'major_class_percent_{process_stage}', major_class_percent)

    def forward(self, input_ids, attention_mask, score):
#        with torch.no_grad():
#            self.codebert.eval()
        codebert = self.codebert(input_ids, attention_mask)

        codebert_output = self.activation(codebert['pooler_output'])
        dense_input = self.dropout(codebert_output)
        dense = self.dense(dense_input)

        regression_input = self.activation(dense)
        regression = self.regression(regression_input)

        # regression2_input = self.activation(regression)
        # regression2 = self.regression(regression2_input)

        return regression
        # return self.full_regression(codebert_output)

    def training_step(self, batch, batch_idx):
        #print(batch)
        preds = self.forward(**batch).reshape(-1)
        loss = self.loss(preds, batch['score'])
        # loss = (preds - batch['score']) ** 2
        # loss = (preds - batch['score']).abs()
        '''
        print('batch:', batch_idx)
        print('preds', preds)
        print('truth', batch['score'])
        print('loss', loss.mean())
        print()
        '''
        loss = loss.mean()

        # acc = self.score(preds.reshape(-1), batch['score'])

        self.log('train_batch_loss', loss)
        # self.log('train_batch_score', acc)

        self._calculate_major_class(preds, 'train')

        return loss

    def training_epoch_end(self, outputs):
        # self.log('train_acc_epoch', self.score.compute())
        pass

    def validation_step(self, batch, batch_idx):
        preds = self.forward(**batch).reshape(-1)
        # acc = self.score(preds.reshape(-1), batch['score'])
        loss = self.loss(preds, batch["score"])
        self.log('val_batch_loss', loss)
        # self.log('val_batch_score', acc)
        
        return loss
        # return acc

    def validation_epoch_end(self, outputs):
        self.log('val_mean_loss', np.mean([o.item() for o in outputs]))
        # self.log('val_acc_epoch', self.score.compute())
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
        return [optimizer], [lr_scheduler]


class JupyterPairsDataModule(pl.LightningDataModule):

    def __init__(
            self, train_path: str, test_path: str = None, model="microsoft/codebert-base",
            train_val_split: float = 0.8, batch_size: int = 32, padding: int = 512,
            make_abs: bool = False, binarize: bool = False
        ):

        super().__init__()

        self.test_path = test_path
        self.train_path = train_path

        self.batch_size = batch_size
        self.padding = padding

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model)
        self.train_val_split = train_val_split

        self.train_dataset = None
        self.test_dataset = None

        self.make_abs = make_abs
        self.binarize = binarize

    def prepare_data(self):
        print('____Preparing train dataset____')
        self.train_dataset = self._prepare_dataset()
        print(self.train_dataset)
        # print('____Prepearing test dataset____')
        # self.test_dataset = self._prepare_dataset(test = True)
        print('____Preparations finished____')

    def _prepoces_dataset(self, df, test=False):

        df = df.sample(100000)

        df = df.replace(r'[^A-Za-z0-9\s]+', ' ', regex=True)
        df = df.replace(r'\s+', ' ', regex=True)

        df.loc[(df.p1 == ' ') | (df.p1 == ''), 'p1'] = pd.NA
        df.loc[(df.p2 == ' ') | (df.p2 == ''), 'p2'] = pd.NA

        df = df.dropna()

        df.loc[:, 'source'] = df.loc[:, 'p1'] + ' </s> ' + df.loc[:, 'p2']
        df = df.loc[:, ['source', 'score']]

        if self.make_abs:
            df.loc[:, 'score'] = np.abs(df.loc[:, 'score'])

        if self.binarize:
            df.loc[:, 'score'] = np.isclose(np.round(df.loc[:, 'score'].abs(), 6), 0.262364).astype(float)

        return df

    def _prepare_dataset(self, test=False):

        path = self.test_path if test else self.train_path
        df = pd.read_feather(path)

        df = self._prepoces_dataset(df)

        dataset = Dataset.from_pandas(df)

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

        dataset.set_format('pt', ['input_ids', 'attention_mask', 'score'])

        return dataset

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.train_dataset = self.train_dataset.train_test_split(test_size=0.1)

        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset['train'], batch_size=self.batch_size, num_workers=24, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset['test'], batch_size=self.batch_size, num_workers=24, pin_memory=True)

    
JDM = JupyterPairsDataModule(f'data/log_symmetric_pairs.fth', batch_size=32, binarize=True)

wandb_logger = WandbLogger(project="JupyterBert", entity="jbr_jupyter")

trainer = pl.Trainer(accelerator="gpu",
                     max_epochs=10,
                     logger=wandb_logger,
                     devices=[2],
                     enable_progress_bar=True,
                     log_every_n_steps=1,
                     accumulate_grad_batches=4,
                     val_check_interval=0.5)
                     


model = CodeBERTBaseLine()
trainer.fit(model, JDM)
