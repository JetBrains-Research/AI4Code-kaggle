from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from argparse import ArgumentParser
from datasets import Dataset

tqdm.pandas()


def get_code_subsample(group, sep_token_id, n=20, max_len=19):
    code_df = group[group.cell_type == "code"]

    n = len(code_df) if n > len(code_df) else n

    if n == len(code_df):
        selection = code_df["tokens"]
    else:
        indices = np.random.choice(len(code_df) - 2, n - 2, replace=False) + 1
        indices = [0] + sorted(indices.tolist()) + [len(code_df) - 1]
        selection = code_df["tokens"].iloc[indices]
    tokens = selection.apply(
        lambda tokens: tokens[:max_len - 1] + [sep_token_id]
    )
    tokens = np.concatenate(tokens.values).tolist()

    return tokens


def build_attention_masks(row, md_len, code_len):
    n_md = len(row["tokens"]) + 2
    n_md = min(n_md, md_len)

    n_code = len(row["code_subsample"])
    n_code = min(n_code, code_len)

    mask = (
            [1] * n_md +
            [0] * (md_len - n_md) +
            [1] * n_code +
            [0] * (code_len - n_code)
    )
    return mask


def build_input_ids(row, cls_token_id, sep_token_id, pad_token_id, md_len, code_len):
    n_md = len(row["tokens"]) + 2
    n_md = min(n_md, md_len)

    n_code = len(row["code_subsample"])
    n_code = min(n_code, code_len)

    input_ids = (
            [cls_token_id] +
            row["tokens"][:md_len - 2] +
            [sep_token_id] +
            [pad_token_id] * (md_len - n_md) +
            row["code_subsample"][:code_len] +
            [pad_token_id] * (code_len - n_code)
    )

    return input_ids


def main(args):
    df = pd.read_feather(args.input_fth, columns=["id", "cell_id", "cell_type", args.model, "score"])
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    df["tokens"] = df[args.model].progress_apply(lambda tokens: list(map(int, tokens.split())))
    code_subsamples = df.groupby("id").apply(
        lambda group: get_code_subsample(group, tokenizer.sep_token_id)
    )
    df = df.merge(code_subsamples.rename("code_subsample"), on="id", how="left")
    md_mask = df.cell_type == "markdown"

    df["attention_mask"] = None
    df.loc[md_mask, "attention_mask"] = df[md_mask].progress_apply(
        lambda row: build_attention_masks(row, 128, 19 * 20),
        axis=1
    )

    df["input_ids"] = None
    df.loc[md_mask, "input_ids"] = df[md_mask].progress_apply(
        lambda row: build_input_ids(
            row,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            128,
            19 * 20
        ),
        axis=1
    )

    md_count = df[df.cell_type == "markdown"].groupby("id").size()
    code_count = df[df.cell_type == "code"].groupby("id").size()
    df = df.merge(md_count.rename("md_count"), on="id", how="left")
    df = df.merge(code_count.rename("code_count"), on="id", how="left")

    pt_cols = ["input_ids", "attention_mask", "score", "md_count", "code_count"]
    str_cols = ["id", "cell_id"]
    df = df[pt_cols + str_cols]
    df = df[md_mask]
    df = df.reset_index(drop=True)
    df.to_feather(args.output_fth)

    dataset = Dataset.from_pandas(df)
    dataset.set_format('pt', pt_cols, output_all_columns=True)
    dataset = dataset.rename_column('id', 'notebook_id')
    dataset.save_to_disk(args.output_dat)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_fth", type=str, help=".fth with dataframe for processing")
    parser.add_argument("output_fth", type=str, help=".fth to store the processed dataframe")
    parser.add_argument("output_dat", type=str, help=".dat to store the dataset")
    parser.add_argument("model", type=str, help="Model from HuggingFace to use")
    args = parser.parse_args()

    main(args)
