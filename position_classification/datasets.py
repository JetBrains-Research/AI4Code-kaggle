import pandas as pd
import torch
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = [label for label in df[df['cell_type'] == 'markdown']['position_class']]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df[df['cell_type'] == 'markdown']['clean_text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return torch.tensor(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
