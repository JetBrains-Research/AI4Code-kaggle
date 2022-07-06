import pandas as pd
from pathlib import Path
from tqdm import tqdm

from preprocessing import MdProcessor

pd.options.display.width = 180
pd.options.display.max_colwidth = 120

data_dir = Path(r'data\ai4code')
NUM_TRAIN = 100


def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


if __name__ == '__main__':
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

    md_processor = MdProcessor()
    md_mask = (df['cell_type'] == 'markdown')
    df['processed_source'] = None
    df.loc[md_mask, 'processed_source'] = df[md_mask].source.apply(lambda row: md_processor.process(row))

    print(df[df.cell_type == 'markdown'].processed_source.iloc[300])
