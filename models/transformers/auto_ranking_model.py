import torch
from transformers import DistilBertModel

from .abstract_ranking_model import AbstractRankingModel


class AutoRankingModel(AbstractRankingModel):
    def __init__(self, model="distilbert-base-uncased"):
        super(AutoRankingModel, self).__init__()

        self.distill_bert = DistilBertModel.from_pretrained(model, return_dict=True)
        self.dense = torch.nn.Linear(768, 1)
        self.loss = torch.nn.MSELoss()
        self.activation = torch.nn.LeakyReLU()

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        embeddings = self.distill_bert(input_ids, attention_mask)['last_hidden_state']
        embeddings = self.activation(embeddings)
        preds = self.dense(embeddings[:, 0, :])  # why are you taking embeding of first token, maybe mean?

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01
        )
        return optimizer
