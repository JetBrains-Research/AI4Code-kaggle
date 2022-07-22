import torch
from transformers import DistilBertModel, AutoModel, get_linear_schedule_with_warmup

from .abstract_ranking_model import AbstractRankingModel


class AutoRankingModel(AbstractRankingModel):
    def __init__(
            self,
            # learning_rate=1e-5,
            model="distilbert-base-uncased",
            optimizer_config=None,
            scheduler_config=None,
    ):
        super(AutoRankingModel, self).__init__()

        self.features = [
            "normalized_plot_functions",
            "normalized_sloc",
            "code_count",
            "md_count",
            "normalized_defined_functions"
        ]

        self.distill_bert = AutoModel.from_pretrained(model, return_dict=True)
        self.dense = torch.nn.Linear(768, 1)

        self.loss = torch.nn.MSELoss()
        self.activation = torch.nn.LeakyReLU()

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.optimizer = None
        self.scheduler = None

        # self.learning_rate = learning_rate

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        embeddings = self.distill_bert(input_ids, attention_mask)['last_hidden_state']
        embeddings = self.activation(embeddings)
        embeddings = embeddings[:, 0, :]

        preds = self.dense(embeddings)  # why are you taking embedding of first token, maybe mean?

        return preds

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            # lr=self.learning_rate,
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
