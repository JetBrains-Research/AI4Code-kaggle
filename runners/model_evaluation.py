import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_managment.preprocessing import kaggle_cleaning
from runners.order_builder import OrderBuilder


class PairwiseDataset(Dataset):

    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df) ** 2

    def __getitem__(self, ind):
        i = ind // len(self.df)
        j = ind % len(self.df)
        tokens = (
            [self.tokenizer.cls_token_id]
            + self.df.iloc[i]["text_tokenized"]
            + [self.tokenizer.sep_token_id]
            + self.df.iloc[j]["text_tokenized"]
            + [self.tokenizer.sep_token_id],
        )
        return tokens

    def collate_fn(self, batch):
        tokens = [{'input_ids': x} for x in batch]
        res = self.tokenizer.pad(
            tokens, return_attention_mask=True, return_tensors="pt"
        )
        return {
            'input_ids': res['input_ids'],
            'attention_mask': res['attention_mask']
        }


class PairwiseKendallTauEvaluator:

    def __init__(self, path_to_df, tokenizer, max_p_length):
        df = pd.read_feather(path_to_df)
        df["source"] = df["source"].apply(lambda s: kaggle_cleaning(s))
        df["text_tokenized"] = df["source"].apply(
            lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][:max_p_length]
        )

        self.df = df
        self.tokenizer = tokenizer
        self.notebooks = df.id.unique()

    def evaluate(self, model, batch_size, device):
        total_inv = 0
        total_max_inv = 0
        with tqdm(self.notebooks) as pbar:
            for notebook in pbar:
                cells = self.df.loc[self.df.id == notebook]
                n_cells = len(cells)
                is_code = (cells.cell_type == "code").values

                dataset = PairwiseDataset(cells, self.tokenizer)
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn
                )
                all_preds = []
                for batch in data_loader:

                    for k, v in batch.items():
                        batch[k] = v.to(device)

                    with torch.no_grad():
                        preds = model.forward(batch)
                        preds = preds.exp()
                        preds = preds[:, 1] / preds.sum(-1)
                        all_preds.append(preds)

                probs = torch.cat(all_preds).view(n_cells, n_cells)
                probs[torch.arange(n_cells), torch.arange(n_cells)] = 0

                order = OrderBuilder.greedy_pairwise(probs, is_code)

                true_order = torch.arange(n_cells)

                inv, max_inv = OrderBuilder.kendall_tau(true_order.tolist(), order)
                kt = 1 - 4 * inv / max_inv

                total_inv += inv
                total_max_inv += max_inv

                pbar.set_postfix(kendall_tau=1 - 4 * total_inv / total_max_inv, cur_kendall_tau=kt)

        kt = 1 - 4 * total_inv / total_max_inv
        return kt
