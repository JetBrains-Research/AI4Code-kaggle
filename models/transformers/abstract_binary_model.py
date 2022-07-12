import numpy as np
import torch
import pytorch_lightning as pl
from abc import abstractmethod, ABC

from .utils import extract_value


class AbstractBinaryModel(pl.LightningModule, ABC):
    def __init__(self, evaluator=None):
        super(AbstractBinaryModel, self).__init__()
        self.evaluator = evaluator
        self.loss_function = torch.nn.CrossEntropyLoss()

    def _calculate_major_class(self, x, process_stage):
        major_class_percent = np.around(
            x.unique(return_counts=True)[1].max().cpu(), 5
        ) / len(x)
        self.log(f"major_class_percent_{process_stage}", major_class_percent)

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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self._shared_step(batch, batch_idx, f"val_{dataloader_idx}")

    def validation_epoch_end(self, outputs):
        for idx, val_outputs in enumerate(outputs):
            self._shared_epoch_end(val_outputs, f"val_{idx}")
        if self.evaluator is not None:
            kendall_tau = self.evaluator(self, self.trainer.val_dataloaders[0].batch_size, self.device)
            self.log("val/kendall_tau", kendall_tau)

    @staticmethod
    def _compute_metrics(preds, batch, stage):
        log = {}

        preds = preds.exp()
        preds = preds[:, 1] / preds.sum(-1)
        binarized_preds = preds >= 0.5
        true_positive = (
            (batch["score"].view(-1) == 1)
            & (binarized_preds == batch["score"].view(-1))
        ).sum()
        true_negative = (
            (batch["score"].view(-1) == 0)
            & (binarized_preds == batch["score"].view(-1))
        ).sum()
        all_score = batch["score"].sum()
        all_preds = binarized_preds.sum()

        recall = true_positive / all_score if all_score else 0.0
        precision = true_positive / all_preds if all_preds else 0.0
        f1 = (
            2 * recall * precision / (recall + precision) if recall + precision else 0.0
        )
        acc = (true_positive + true_negative) / len(batch["score"])

        log[f"{stage}/recall"] = recall
        log[f"{stage}/precision"] = precision
        log[f"{stage}/f1"] = f1
        log[f"{stage}/acc"] = acc

        return log

    def _shared_step(self, batch, batch_idx, stage):
        preds = self.forward(batch)
        loss = self.loss_function(preds, batch["score"].view(-1))
        log = self._compute_metrics(preds, batch, stage)

        log[f"{stage}/loss"] = loss

        if stage == "train":
            self.log_dict(log, on_step=True, on_epoch=False)

        log["loss"] = loss

        return log

    def _shared_epoch_end(self, outputs, stage):
        log = {}
        for metric in outputs[0]:
            if metric == "loss":
                continue
            values = [extract_value(o[metric]) for o in outputs]
            log[f"epoch_{metric}"] = np.mean(values)

        self.log_dict(log, on_step=False, on_epoch=True)
