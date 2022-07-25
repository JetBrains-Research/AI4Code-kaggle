import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser

from transformers import AutoTokenizer

from data_managment.preprocessing import DatasetProcessor

tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("path", type=str, help=".fth with dataframe for processing")
args = parser.parse_args()


df = pd.read_feather(args.path)
dp = DatasetProcessor(df)
df = dp.process_dataset()
df.to_feather(args.path)
