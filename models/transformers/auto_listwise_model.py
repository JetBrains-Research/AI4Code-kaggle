import pytorch_lightning as pl
from transformers import AutoModel, get_linear_schedule_with_warmup
import torch
from collections import defaultdict
import numpy as np

from models.transformers.utils import extract_value
from prepare_listwise_dataset_constants import *
from runners.order_builder import OrderBuilder


class ListwiseModel(pl.LightningModule):

    def __init__(
            self,
            model_name,
            md_len,
            md_total,
            optimizer_config=None,
            scheduler_config=None,
            code_ids=None,
    ):
        super(ListwiseModel, self).__init__()
        
        self.md_len = md_len
        self.md_total = md_total
        
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(768, 1)

        self.loss_function = torch.nn.L1Loss()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.optimizer = None
        self.scheduler = None

        self.notebook_predictions = self._clean_notebook_predictions()
        self.built_orders = {}
        self.code_ids = code_ids

    @staticmethod
    def _clean_notebook_predictions():
        return defaultdict(lambda: defaultdict(list))

    def forward(self, batch):
        # BATCH_SIZE x SEQ_LEN
        input_ids = batch['input_ids']
        # BATCH_SIZE x SEQ_LEN
        attention_mask = batch['attention_mask']

        batch_size = len(input_ids)

        # BATCH_SIZE x SEQ_LEN x EMB_SIZE
        embeddings = self.model(input_ids, attention_mask)[0]
        # BATCH_SIZE x SEQ_LEN x EMB_SIZE
        embeddings = embeddings * attention_mask.unsqueeze(-1)

        # BATCH_SIZE x N_MD x MD_LEN x EMB_SIZE
        md_embeddings = embeddings[:, :self.md_total, :].view(batch_size, -1, self.md_len, EMB_SIZE)

        # BATCH_SIZE x N_MD
        md_counts = attention_mask[:, :self.md_total].view(batch_size, -1, self.md_len).sum(-1)

        # BATCH_SIZE x N_MD x EMB_SIZE
        md_embeddings = md_embeddings.sum(-2) / (md_counts.unsqueeze(-1) + 1e-9)

        # BATCH_SIZE x N_MD x 1
        predictions = self.linear(md_embeddings)

        return predictions.squeeze(-1)

    def training_step(self, batch, batch_idx):
        log = self._shared_step(batch, batch_idx, "train")

        if self.scheduler is not None:
            self.scheduler.step()
            self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)

        return log

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def _shared_step(self, batch, batch_idx, stage):
        pred = self(batch)
        gt = batch["score"]
        #         loss_mask = batch["score_mask"]

        #         loss = ((pred - gt) * loss_mask).abs().sum() / loss_mask.sum()
        loss = self.loss_function(pred, gt)

        if stage == "train":
            self.log(f"{stage}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        if stage != "train":

            for preds, cell_ids, notebook_id in zip(pred, batch["cell_ids"], batch["notebook_id"]):
                if len(preds) == 1:
                    cell_ids = [cell_ids]
                    
                for val, cell_id in zip(preds, cell_ids):
                    if not isinstance(cell_id, str):
                        cell_id = int(cell_id)
                    self.notebook_predictions[notebook_id][cell_id].append(val)

        return {"loss": loss}

    def _shared_epoch_end(self, outputs, stage):
        #         print(outputs)

        log = {
            f"{stage}/loss": np.mean([extract_value(x["loss"]) for x in outputs])
        }

        self.log_dict(log, on_step=False, on_epoch=True)

        if stage != "train":

            orders = {}

            for n_id, predictions in self.notebook_predictions.items():
                if self.code_ids is None:
                    n_code = ... # TODO: fix
                else:
                    code_ids = self.code_ids[n_id]
                    n_code = len(code_ids)

                cell_ids = list(predictions.keys())

                if isinstance(cell_ids[0], str):
                    code_ranks = [((i + 1) / (n_code + 1), i) for i in range(n_code)]
                else:
                    code_ranks = [((i + 1) / (n_code + 1), ind) for i, ind in enumerate(code_ids)]

                md_ranks = [(
                    np.mean([
                        extract_value(x) for x in preds
                    ]), cell_id)
                    for cell_id, preds in predictions.items()
                ]
                ranks = sorted(code_ranks + md_ranks)
                order = [cell_id for _, cell_id in ranks]
                orders[n_id] = order


            self.built_orders = orders
            # kt = OrderBuilder.evaluate_notebooks(orders)
            total_inv, total_max_inv = 0, 0
            for n_id, order in orders.items():
                true_order = sorted(order)
                inv, max_inv = OrderBuilder.kendall_tau(true_order, order)
                total_inv += inv
                total_max_inv += max_inv
            kt = 1 - 4 * total_inv / total_max_inv
            self.log(f"{stage}_kendall_tau", kt, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.optimizer_config,
        )

        if self.scheduler_config is not None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.scheduler_config["warmup_steps"],
                num_training_steps=self.scheduler_config["training_steps"],
            )
            print(f"Skipping {self.scheduler_config['cur_step']} steps in scheduler")
            for _ in range(self.scheduler_config['cur_step']):
                self.scheduler.step()

        return self.optimizer


class CleanListwisePredictionCallback(pl.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.notebook_predictions = pl_module._clean_notebook_predictions()
        pl_module.built_orders = {}
