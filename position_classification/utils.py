import nltk
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from preprocessing import MdProcessor

nltk.download('stopwords')
pd.options.display.width = 180
pd.options.display.max_colwidth = 120

data_dir = Path(r'C:\Users\Konstantin\Documents\datasets\ai4code')
NUM_TRAIN = 500


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


def get_position(a):
    if a < 0.33:
        return 0
    elif 0.33 <= a < 0.66:
        return 1
    else:
        return 2


def prepare_df():
    paths_train = list((data_dir / 'train').glob('*.json'))[:NUM_TRAIN]
    notebooks_train = [
        read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
    ]
    df = (
        pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )

    df_orders = pd.read_csv(
        data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()  # Split the string representation of cell_ids into a list


    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right',
    )

    ranks = {}
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
    )

    df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
    df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
    df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

    md_processor = MdProcessor()
    md_mask = (df['cell_type'] == 'markdown')
    df['processed_source'], df['clean_text'] = None, None
    df.loc[md_mask, 'processed_source'] = df[md_mask].source.apply(lambda row: md_processor.process(row))
    df.loc[md_mask, 'clean_text'] = df[md_mask].processed_source.apply(lambda row: json.loads(row).get('text'))
    df['position_class'] = df.pct_rank.apply(lambda a: get_position(a))

    return df
