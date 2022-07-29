import numpy as np
import torch
from dataclasses import dataclass
from tqdm.auto import tqdm

@dataclass
class ProcessedNotebook:
    code_tokens: list
    code_scores: list
    md_tokens: torch.tensor
    md_scores: torch.tensor

    md_cell_ids: np.ndarray

    n_md: int
    n_code: int
    notebook_id: str = ""

        

class NotebookDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        datapoints,
        sep_token_id,
        pad_token_id,
        md_len,
        code_len,
        total_md_len,
        total_code_len,
    ):
        self.datapoints = datapoints
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        
        self.md_len = md_len
        self.code_len = code_len
        
        self.total_md_len = total_md_len
        self.total_code_len = total_code_len
        self.total_len = total_md_len + total_code_len

        self.select_md = self.total_md_len // self.md_len
        self.select_code = self.total_code_len // self.code_len

        self.reps = self.select_md

        self.n_examples = 0
        for i, datapoint in tqdm(enumerate(datapoints)):
            
            n_md = len(datapoint.md_tokens)
            md_tokens = torch.full((n_md, self.md_len), self.pad_token_id, dtype=torch.long)
            for i, token_string in enumerate(datapoint.md_tokens):
                if isinstance(token_string, str):
                    for j, x in enumerate(token_string.split()[:md_len]):
                        md_tokens[i, j] = int(x)
                else:
                    md_tokens[i, :] = token_string
            datapoint.md_tokens = md_tokens
            
            n_code = len(datapoint.code_tokens)
            code_tokens = torch.full((n_code, self.code_len), self.pad_token_id, dtype=torch.long)
            for i, token_string in enumerate(datapoint.code_tokens):
                if isinstance(token_string, str):
                    for j, x in enumerate(token_string.split()[:code_len]):
                        code_tokens[i, j] = int(x)
                else:
                    code_tokens[i, :] = token_string
                    
            datapoint.code_tokens = code_tokens
            
            self.n_examples += n_md

        self.notebook_indices = torch.zeros(self.n_examples, dtype=torch.int)
        cur_len = 0
        for i, datapoint in enumerate(datapoints):
            n_md = datapoint.md_tokens.size(0)
            self.notebook_indices[cur_len:cur_len + n_md] = i
            cur_len += n_md

        self.selected_permutations = torch.zeros(self.n_examples, self.select_md, dtype=torch.long)
        self.reset_dataset()

    def __len__(self):
        return self.n_examples

    def reset_dataset(self):
        cur_len = 0
        for i, datapoint in enumerate(self.datapoints):
            n_md = datapoint.md_tokens.size(0)
            self.selected_permutations[cur_len:cur_len + n_md, :] = torch.cat([
                torch.randperm(n_md) for _ in range(self.reps)
            ]).view(-1, self.select_md)
            cur_len += n_md

    @staticmethod
    def select_n(tokens, scores, max_len, keep_order):
        n_tokens = tokens.size(0)
        len_tokens = tokens.size(1)

        n_selected = max_len // len_tokens

        if n_selected >= n_tokens:
            if keep_order:
                return tokens, scores
            else:
                indices = torch.randperm(n_tokens)
                return tokens[indices], scores[indices]

        if keep_order:
            middle_inds = np.random.choice(n_tokens - 2, n_selected - 2, replace=False)
            middle_inds.sort()
            indices = torch.cat((
                torch.tensor([0]),
                torch.tensor(middle_inds + 1),
                torch.tensor([n_tokens - 1])
            ))
            return tokens[indices], scores[indices]
        else:
            indices = torch.randperm(n_tokens)[:n_selected]
            return tokens[indices], scores[indices]

    def __getitem__(self, ind):
        notebook_ind = self.notebook_indices[ind]
        datapoint = self.datapoints[notebook_ind]
        permutation = self.selected_permutations[ind]

        code_tokens, code_scores = self.select_n(
            datapoint.code_tokens, datapoint.code_scores, self.total_code_len, True
        )

        md_tokens = datapoint.md_tokens[permutation]
        md_scores = datapoint.md_scores[permutation]
        md_cell_ids = datapoint.md_cell_ids[permutation]

        input_ids = torch.full((self.total_len,), self.pad_token_id)
        input_ids[:md_tokens.numel()] = md_tokens.view(-1)
        input_ids[self.total_md_len:self.total_md_len + code_tokens.numel()] = code_tokens.view(-1)

        attention_mask = (input_ids != self.pad_token_id).type(torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'score': md_scores,
            'cell_ids': md_cell_ids,
            'notebook_id': datapoint.notebook_id,
        }
    

def collate_fn(batch):
    input_ids = torch.stack([
        x['input_ids'] for x in batch
    ])
    attention_mask = torch.stack([
        x['attention_mask'] for x in batch
    ])
    score = torch.stack([
        x['score'] for x in batch
    ])

    cell_ids = [x['cell_ids'] for x in batch]
    notebook_id = [x['notebook_id'] for x in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'score': score,
        'cell_ids': cell_ids,
        'notebook_id': notebook_id,
    }
