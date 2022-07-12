import pytorch_lightning as pl
import torch
from transformers import DistilBertModel


class MarkdownModelPl(pl.LightningModule):
    def __init__(self, model="distilbert-base-uncased"):
        super(MarkdownModelPl, self).__init__()

        self.distill_bert = DistilBertModel.from_pretrained(model, return_dict=True)
        self.dense = torch.nn.Linear(768, 1)
        self.loss = torch.nn.MSELoss()
        self.activation = torch.nn.LeakyReLU()

    def forward(self, input_ids, attention_mask, score):
        embeddings = self.distill_bert(input_ids, attention_mask)['last_hidden_state']
        embeddings = self.activation(embeddings)
        preds = self.dense(embeddings[:, 0, :])  # why are you taking embeding of first token, maybe mean?

        return preds

    def training_step(self, batch, batch_idx):
        preds = self.forward(**batch).reshape(-1)
        loss = self.loss(preds, batch['score'])
        self.log('train_batch_loss', loss)
        self.log('train_RMSE', 1)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.forward(**batch).reshape(-1)
        loss = self.loss(preds, batch['score'])

        self.log('val_batch_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=3e-4, betas=(0.9, 0.999), eps=1e-08)
        return optimizer
