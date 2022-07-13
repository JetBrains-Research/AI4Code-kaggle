import numpy as np
import torch
import pytorch_lightning as pl
from abc import abstractmethod, ABC

from runners.order_builder import OrderBuilder
from .utils import extract_value


class AbstractRankingModel(pl.LightningModule, ABC):
    def __init__(self):
        super(AbstractRankingModel, self).__init__()
        self.loss_function = torch.nn.L1Loss()

    @abstractmethod
    def forward(self, batch):
        """
        Predict logits for each class. For 2 classes we get 2 outputs per sample.
        :param batch: input batch from the data loader
        :return: predicted logits in form B x 2
        """
        pass

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def _shared_step(self, batch, batch_idx, stage):
        preds = self.forward(batch)
        loss = self.loss_function(preds.view(-1), batch["score"].view(-1))
        log = {f"{stage}/loss": loss}

        if stage == "train":
            self.log_dict(log, on_step=True, on_epoch=False)

        log["loss"] = loss
        log["batched_predictions"] = (
            preds.view(-1),
            batch["score"].view(-1),
            batch["md_count"],
            batch["code_count"],
            batch["notebook_id"],
        )

        return log

    def _shared_epoch_end(self, outputs, stage):
        log = {}
        for metric in outputs[0]:
            if metric == "loss" or metric == "batched_predictions":
                continue
            values = [extract_value(o[metric]) for o in outputs]
            log[f"epoch_{metric}"] = np.mean(values)

        batched_preds = [output["batched_predictions"] for output in outputs]

        preds = torch.cat([x[0] * (x[2] + x[3]) for x in batched_preds]).round()
        scores = torch.cat([x[1] * (x[2] + x[3]) for x in batched_preds]).round()
        md_counts = torch.cat([x[2] for x in batched_preds])
        code_counts = torch.cat([x[3] for x in batched_preds])

        total_inv, total_max_inv = 0, 0

        notebook_ids = np.concatenate([x[-1] for x in batched_preds])
        for notebook_id in np.unique(notebook_ids):
            loc = notebook_id == notebook_ids
            pred_positions = preds[loc]
            true_positions = scores[loc]
            n_md = md_counts[loc][0].item()
            n_code = code_counts[loc][0].item()

            true_order = OrderBuilder.greedy_ranked(true_positions, n_md, n_code)
            pred_order = OrderBuilder.greedy_ranked(pred_positions, n_md, n_code)

            inv, max_inv = OrderBuilder.kendall_tau(true_order, pred_order)
            total_inv += inv
            total_max_inv += max_inv

        kt = 1 - 4 * total_inv / total_max_inv
        log[f"{stage}_kendall_tau"] = kt

        self.log_dict(log, on_step=False, on_epoch=True)
