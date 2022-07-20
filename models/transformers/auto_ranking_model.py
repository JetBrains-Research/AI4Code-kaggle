import torch
from transformers import DistilBertModel, AutoModel

from .abstract_ranking_model import AbstractRankingModel
            
class AutoRankingModel(AbstractRankingModel):
    def __init__(
            self,
#             learning_rate=1e-5,
            model="distilbert-base-uncased",
            optimizer_config=None,
            scheduler_config=None,
            dropout_rate=0.0,
            test_notebook_order=None,
            hidden_size=100,
            use_features=False
    ):
        super(AutoRankingModel, self).__init__(test_notebook_order=test_notebook_order)

        self.features = [
            "normalized_plot_functions",
            "normalized_sloc",
            "code_count",
            "md_count",
            "normalized_defined_functions"
        ]
        self.use_features = use_features

        self.distill_bert = AutoModel.from_pretrained(model, return_dict=True)
        if not use_features:
            self.dense = torch.nn.Linear(768, 1)
        else:
            self.dense_1 = torch.nn.Linear(768 + len(self.features), hidden_size)
#             self.dense_2 = torch.nn.Linear(hidden_size + len(self.features), 1)
            self.dense_2 = torch.nn.Linear(hidden_size, 1)

        self.loss = torch.nn.MSELoss()
        self.activation = torch.nn.LeakyReLU()
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler = None

#         self.learning_rate = learning_rate

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        embeddings = self.dropout(self.distill_bert(input_ids, attention_mask)['last_hidden_state'])
        embeddings = self.activation(embeddings)
        embeddings = embeddings[:, 0, :]

        if not self.use_features:
            preds = self.dense(embeddings)  # why are you taking embedding of first token, maybe mean?
        else:
            features = torch.cat([
                batch[feat].unsqueeze(-1)
                for feat in self.features
            ], dim=-1)
            embeddings = torch.cat([embeddings, features], dim=-1)
            hidden_state = self.activation(self.dense_1(embeddings))
#             hidden_state = torch.cat([hidden_state, features], dim=-1)
            preds = self.dense_2(hidden_state)

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
#             lr=self.learning_rate,
            **self.optimizer_config,
        )
        
        if self.scheduler_config is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                **self.scheduler_config,
            )
            # self.scheduler = torch.optim.lr_scheduler.StepLR(
            #    optimizer,
            #    **self.scheduler_config
            # )
            # return [optimizer], [self.scheduler]
        
        return optimizer
        
