import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser

from transformers import AutoTokenizer

tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("path", type=str, help=".fth with dataframe for processing")
parser.add_argument("model", type=str, help="Model from HuggingFace to use")
args = parser.parse_args()


def tokenize_doc(doc, tokenizer):
    tokenized = tokenizer(
        doc[:1024],
        padding=False,
        truncation=True,
        max_length=512,
        add_special_tokens=False,
        return_attention_mask=False
    )
    return " ".join(str(ind) for ind in tokenized['input_ids'])


df = pd.read_feather(args.path)
tokenizer = AutoTokenizer.from_pretrained(args.model)
df[args.model] = df['cleaned_source'].progress_map(
    lambda doc: tokenize_doc(doc, tokenizer),
)
df.save_feather(args.path)
