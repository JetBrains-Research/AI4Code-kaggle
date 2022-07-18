import torch
from transformers import DistilBertModel, AutoModel

from .abstract_ranking_model import AbstractRankingModel
            
class AutoRankingModel(AbstractRankingModel):
    def __init__(
            self,
            learning_rate=1e-5,
            model="distilbert-base-uncased",
            optimizer_config=None,
            scheduler_config=None,
            dropout_rate=0.0,
            test_notebook_order=None,
    ):
        super(AutoRankingModel, self).__init__(test_notebook_order=test_notebook_order)

        self.distill_bert = AutoModel.from_pretrained(model, return_dict=True)
        self.dense = torch.nn.Linear(768, 1)
        self.loss = torch.nn.MSELoss()
        self.activation = torch.nn.LeakyReLU()
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler = None
        
        self.learning_rate = learning_rate

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        embeddings = self.dropout(self.distill_bert(input_ids, attention_mask)['last_hidden_state'])
        embeddings = self.activation(embeddings)
        preds = self.dense(embeddings[:, 0, :])  # why are you taking embeding of first token, maybe mean?

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            **self.optimizer_config,
        )
        
        if self.scheduler_config is not None:
#             self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
#                 optimizer,
#                 **self.scheduler_config,
#             )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
               optimizer,
               **self.scheduler_config 
            )
            return [optimizer], [self.scheduler]
        
        return optimizer
        
