import torch
from transformers import AutoModelForSequenceClassification

from .abstract_binary_model import AbstractBinaryModel


class AutoBinaryModel(AbstractBinaryModel):
    def __init__(self, model_name):
        super(AutoBinaryModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        return self.model(input_ids, attention_mask).logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer
