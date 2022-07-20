import json
import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from markdown import markdown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from tqdm import tqdm

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')

tqdm.pandas()

stemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def kaggle_cleaning(document, min_len=4):
    document = re.sub(r"_", " ", document)
    document = re.sub(r"\W", " ", document)
    document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)
    document = re.sub(r"\^[a-zA-Z]\s+", " ", document)
    document = re.sub(r"\s+", " ", document)
    document = re.sub(r"^b\s+", " ", document)
    document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)

    document = document.strip().lower()
    tokens = document.split()
    tokens = [stemmer.lemmatize(token) for token in tokens]
    tokens = [
        token
        for token in tokens
        if len(token) >= min_len and token not in stop_words
    ]
    return " ".join(tokens)


def clean_text(document):
    # Remove all the special characters
    document = re.sub(r"\W", " ", str(document))
    # remove all single characters
    document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)
    # Remove single characters from the start
    document = re.sub(r"\^[a-zA-Z]\s+", " ", document)
    # Substituting multiple spaces with single space
    document = re.sub(r"\s+", " ", document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r"^b\s+", "", document)
    # Converting to Lowercase
    document = document.lower()
    return document


class MdProcessor:
    def __init__(self):
        self.features = {
            "text": lambda s: clean_text("".join(s.findAll(text=True))),
            "headers": self.rule2text(re.compile("^h[1-6]$")),
            "bold_text": self.rule2text("strong"),
            "italic_text": self.rule2text("i"),
            "code": self.rule2text("code"),
            "links": self.rule2text("a"),
        }

    @staticmethod
    def rule2text(search_fun):
        return lambda s: [i.text for i in s.find_all(search_fun)]

    def process(self, md_string, nb_index=None, cell_index=None):
        soup = BeautifulSoup(markdown(md_string), "html.parser")
        res = {}

        for name, fun in self.features.items():
            value = fun(soup)
            if isinstance(value, list):
                value = " ".join(value)
            res[name] = value

        if nb_index is not None and cell_index is not None:
            res['id'], res['cell_id'] = nb_index, cell_index

        return res


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_md = df.groupby('cell_type').get_group('markdown').set_index(['id', 'cell_id'])

    md_processor = MdProcessor()
    processed_data = [md_processor.process(row, nb_index, cell_index)
                      for (nb_index, cell_index), row in tqdm(df_md.source.items())]

    processed_data_df = pd.DataFrame(processed_data).set_index(['id', 'cell_id'])
    return df.merge(processed_data_df, on=['id', 'cell_id'], how='left')


class DatasetProcessor:
    def __init__(self, path):
        self.df = pd.read_feather(path)
        self.mapping = {"markdown": MdProcessor}

    @property
    def dataset(self):
        return self.df

    def process_dataset(self):
        for cell_type, processor in self.mapping:
            md_mask = self.df["cell_type"] == cell_type
            self.df["processed_source"] = None
            self.df.loc[md_mask, "processed_source"] = self.df[md_mask].source.apply(
                lambda row: processor.process(row)
            )
