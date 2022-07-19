import numpy as np
import torch
import pytorch_lightning as pl
from abc import abstractmethod, ABC

from runners.order_builder import OrderBuilder
from .utils import extract_value


class AbstractRankingModel(pl.LightningModule, ABC):
    def __init__(self, test_notebook_order=None, ranked_reversed=True):
        super(AbstractRankingModel, self).__init__()
        self.loss_function = torch.nn.L1Loss()
        self.test_notebooks_order = test_notebook_order
        self.ranked_reversed = ranked_reversed

    @abstractmethod
    def forward(self, batch):
        """
        Predict logits for each class. For 2 classes we get 2 outputs per sample.
        :param batch: input batch from the data loader
        :return: predicted logits in form B x 2
        """
        pass

    def training_step(self, batch, batch_idx):
        log = self._shared_step(batch, batch_idx, "train")
#         if self.scheduler is not None:
#             self.scheduler.step()
        return log

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def _shared_step(self, batch, batch_idx, stage):
        preds = self.forward(batch)

        if stage != "test":
            loss = self.loss_function(preds.view(-1), batch["score"].view(-1))
            log = {f"{stage}/loss": loss}

            if stage == "train":
                self.log_dict(log, on_step=True, on_epoch=False)

            log["loss"] = loss
        else:
            log = {}

        if stage != "train":
            log["batched_predictions"] = {
                "preds": preds.view(-1),
                "md_count": batch["md_count"],
                "code_count": batch["code_count"],
                "notebook_id": batch["notebook_id"],
                "cell_id": batch["cell_id"],
            }

            if stage != "test":
                log["batched_predictions"]["score"] = batch["score"].view(-1)

        return log

    def _shared_epoch_end(self, outputs, stage):
        log = {}
        for metric in outputs[0]:
            if metric == "loss" or metric == "batched_predictions":
                continue
            values = [extract_value(o[metric]) for o in outputs]
            log[f"epoch_{metric}"] = np.mean(values)

        if stage == "train":
            self.log_dict(log, on_step=False, on_epoch=True)
            return

        batched_preds = [output["batched_predictions"] for output in outputs]

        if stage != "test":
            scores = torch.cat([x["score"] * (x["md_count"] + x["code_count"]) for x in batched_preds]).round()

        preds = torch.cat([x["preds"] * (x["md_count"] + x["code_count"]) for x in batched_preds]).round()
        md_counts = torch.cat([x["md_count"] for x in batched_preds])
        code_counts = torch.cat([x["code_count"] for x in batched_preds])

        total_inv, total_max_inv = 0, 0
        all_invs, all_max_invs = [], []

        notebook_ids = np.concatenate([x["notebook_id"] for x in batched_preds])
        cell_ids = np.concatenate([x["cell_id"] for x in batched_preds])

        for notebook_id in np.unique(notebook_ids):
            loc = notebook_id == notebook_ids
            pred_positions = preds[loc]
            n_md = md_counts[loc][0].item()
            n_code = code_counts[loc][0].item()
            
            if len(pred_positions) != n_md:
                assert len(outputs) <= 2
                pred_positions = torch.cat([pred_positions, torch.zeros(n_md - len(pred_positions)).to(self.device)])

            if self.ranked_reversed:
                pred_order = OrderBuilder.greedy_ranked_reversed(pred_positions, n_md, n_code)
            else:
                pred_order = OrderBuilder.greedy_ranked(pred_positions, n_md, n_code)
            
            if stage != "test":
                true_positions = scores[loc]
                
                if len(true_positions) != n_md:
                    assert len(outputs) <= 2
                    true_positions = torch.cat([true_positions, torch.zeros(n_md - len(true_positions)).to(self.device)])

                true_order = OrderBuilder.greedy_ranked(true_positions, n_md, n_code)
                inv, max_inv = OrderBuilder.kendall_tau(true_order, pred_order)
                all_invs.append(inv)
                all_max_invs.append(max_inv)
                total_inv += inv
                total_max_inv += max_inv
            else:
                cell_positions = cell_ids[loc]
                for i, prediction in enumerate(pred_order):
                    if prediction >= n_code:
                        pred_order[i] = cell_positions[prediction - n_code]

                self.test_notebooks_order[notebook_id] = pred_order

        if stage != "test":
            all_invs = np.array(all_invs)
            all_max_invs = np.array(all_max_invs)
            inds = np.arange(len(all_invs))
            all_kt = []
            N_BOOTSTRAP = 10_000
            PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
            
            for _ in range(N_BOOTSTRAP):
                inds_subset = np.random.choice(inds, len(inds))
                invs_sum = all_invs[inds_subset].sum()
                max_invs_sum = all_max_invs[inds_subset].sum()
                all_kt.append(1 - 4 * invs_sum / max_invs_sum)
            
            kt = 1 - 4 * total_inv / total_max_inv
            log[f"{stage}_kendall_tau"] = kt
            log[f"{stage}_kendall_tau_std"] = np.std(all_kt)

            all_kt = sorted(all_kt)
            for p in PERCENTILES:
                log[f"{stage}_kendall_tau_{p}p"] = sorted(all_kt)[N_BOOTSTRAP * p // 100]

            self.log_dict(log, on_step=False, on_epoch=True)

