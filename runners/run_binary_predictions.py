import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from models.transformers import AutoBinaryModel
from .model_loader import ModelLoader
from .order_builder import OrderBuilder
from torch.utils.data import Dataset, DataLoader

from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

stemmer = WordNetLemmatizer()

MAX_LEN = 64
BATCH_SIZE = 128

path = ""
model = ModelLoader.load_auto_model(AutoBinaryModel, "")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

df = pd.read_feather(path)

df[["source"]] = df[["source"]].replace(r"\W", " ", regex=True)
df[["source"]] = df[["source"]].replace(r"\s+[a-zA-Z]\s+", " ", regex=True)
df[["source"]] = df[["source"]].replace(r"\^[a-zA-Z]\s+", " ", regex=True)
df[["source"]] = df[["source"]].replace(r"\s+", " ", regex=True)
df[["source"]] = df[["source"]].replace(r"^b\s+", " ", regex=True)
df[["source"]] = df[["source"]].replace(r"\s+[a-zA-Z]\s+", " ", regex=True)


def process(s):
    s = s.strip().lower()
    tokens = s.split()
    tokens = [stemmer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 3]
    return " ".join(tokens)


df["source"] = df["source"].apply(lambda s: process(s))
df["text_tokenized"] = df["source"].apply(
    lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][:MAX_LEN]
)


class JupDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df) ** 2

    def __getitem__(self, ind):
        i = ind // len(self.df)
        j = ind % len(self.df)
        tokens = (
            [tokenizer.cls_token_id]
            + df.iloc[i]["text_tokenized"]
            + [122]
            + df.iloc[j]["text_tokenized"]
            + [tokenizer.sep_token_id],
        )
        return tokens


def collate_fn(batch):
    tokens = [{'input_ids': x} for x in batch]
    res = tokenizer.pad(
        tokens, return_attention_mask=True, return_tensors="pt"
    )
    return {
        'input_ids': res['input_ids'],
        'attention_mask': res['attention_mask']
    }

notebooks = df.id.unique()

total_inv = 0
total_max_inv = 0

with tqdm(notebooks) as pbar:
    for notebook in pbar:
        cells = df.loc[df.id == notebook]
        n_cells = len(cells)
        is_code = cells.cell_type == "code"

        dataset = JupDataset(cells)
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            collate_fn=collate_fn
        )
        all_preds = []
        for batch in data_loader:
            preds = model.forward(batch)
            preds = preds.exp()
            preds = preds[:, 1] / preds.sum(-1)
            all_preds.append(preds)

        probs = torch.cat(all_preds).view(n_cells, n_cells)
        probs[torch.arange(n_cells), torch.arange(n_cells)] = 0

        order = OrderBuilder.greedy_pairwise(probs, is_code)

        order = cells.cell_id.iloc[order]
        print(order)

        true_order = torch.arange(n_cells)

        inv, max_inv = OrderBuilder.kendall_tau(true_order, order)
        total_inv += inv
        total_max_inv += max_inv

        pbar.set_postfix(kendall_tau=1 - 4 * total_inv / total_max_inv)

print(f"Validation kendall tau: {1 - 4 * total_inv / total_max_inv:.5f}")
