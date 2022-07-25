from dataclasses import dataclass

import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from argparse import ArgumentParser
from datasets import Dataset
from prepare_listwise_dataset_constants import *

tqdm.pandas()



@dataclass
class ProcessedNotebook:
    code_tokens: torch.tensor
    code_scores: torch.tensor
    code_len: int
    md_tokens: torch.tensor
    md_scores: torch.tensor
    md_cell_ids: np.ndarray
    md_len: int
    notebook_id: str = ""


def prepare_datapoint(group):
    code_cells = group.cell_type == "code"
    md_cells = group.cell_type == "markdown"

    return ProcessedNotebook(
        code_tokens=torch.tensor(group[code_cells]["trimmed_tokens"].tolist()),
        code_scores=torch.tensor(group[code_cells]["score"].values),
        code_len=len(group[code_cells]["trimmed_tokens"].iloc[0]),
        md_tokens=torch.tensor(group[md_cells]["trimmed_tokens"].tolist()),
        md_scores=torch.tensor(group[md_cells]["score"].values),
        md_cell_ids=group[md_cells]["cell_id"].values,
        md_len=len(group[md_cells]["trimmed_tokens"].iloc[0]),
    )


def trim_tokens(
        group,
        max_total_len,
        cell_type,
        min_len,
        cls_token_id,
        pad_token_id,
        sep_token_id,
        force=False
):
    mask = group.cell_type == cell_type
    n_cells = mask.sum()

    if not force:
        snippet_len = max(max_total_len // n_cells, min_len)
    else:
        snippet_len = min_len

    return group[mask]["tokens"].apply(
        lambda tokens: (
                tokens[:snippet_len - 1] +
                [sep_token_id] +
                [pad_token_id] * max(0, snippet_len - len(tokens) - 1)
        )
    )


def prepare_listwise_dataset(model_name, input_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = pd.read_feather(input_file, columns=["id", "cell_id", "cell_type", model_name, "score"])
    df["tokens"] = df[model_name].progress_apply(lambda tokens: list(map(int, tokens.split())))

    print("Trimming code")
    code_trimmed = df.groupby("id").apply(
        lambda group: trim_tokens(
            group,
            max_total_len=CODE_TOTAL,
            cell_type="code",
            min_len=CODE_LEN,
            cls_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            sep_token_id=tokenizer.sep_token_id,
            force=True
        )
    )
    print("Trimming markdown")
    md_trimmed = df.groupby("id").apply(
        lambda group: trim_tokens(
            group,
            max_total_len=MD_TOTAL,
            cell_type="markdown",
            min_len=MD_LEN,
            cls_token_id=tokenizer.cls_token_id,
            pad_token_id=tokenizer.pad_token_id,
            sep_token_id=tokenizer.sep_token_id,
            force=True,
        )
    )
    df["trimmed_tokens"] = None
    df.loc[df.cell_type == "code", "trimmed_tokens"] = code_trimmed.values
    df.loc[df.cell_type == "markdown", "trimmed_tokens"] = md_trimmed.values

    print("Grouping notebooks")
    datapoints = df.groupby("id").apply(prepare_datapoint)
    for notebook_id, datapoint in datapoints.iteritems():
        datapoint.notebook_id = notebook_id

    return datapoints.values


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_fth", type=str, help=".fth with dataframe for processing")
    # parser.add_argument("output_fth", type=str, help=".fth to store the processed dataframe")
    parser.add_argument("model", type=str, help="Model from HuggingFace to use")
    args = parser.parse_args()

    prepare_listwise_dataset(args.model, args.input_fth)
