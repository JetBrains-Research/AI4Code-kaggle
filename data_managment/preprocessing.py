import re
import json

import pandas as pd
from pandas import DataFrame
from bs4 import BeautifulSoup
from markdown import markdown


def clean_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    return document


class MdProcessor:
    def __init__(self):
        self.features = {
            'text': lambda s: clean_text(''.join(s.findAll(text=True))),
            'headers': self.rule2text(re.compile('^h[1-6]$')),
            'bold_text': self.rule2text('strong'),
            'italic_text': self.rule2text('i'),
            'code': self.rule2text('code'),
            'links': self.rule2text('a'),
        }

    @staticmethod
    def rule2text(search_fun):
        return lambda s: [i.text for i in s.find_all(search_fun)]

    def process(self, md_string):
        soup = BeautifulSoup(markdown(md_string), 'html.parser')
        res = {}
        for name, fun in self.features.items():
            res[name] = fun(soup)

        return json.dumps(res)


def preprocess_dataframe(df: DataFrame) -> DataFrame:
    md_processor = MdProcessor()
    md_mask = (df['cell_type'] == 'markdown')

    df['processed_source'] = None
    df.loc[md_mask, 'processed_source'] = df[md_mask].source.apply(
        lambda row: md_processor.process(row)
    )

    return df


class DatasetProcessor:
    def __init__(self, path):
        self.df = pd.read_feather(path)
        self.mapping = {'markdown': MdProcessor}

    @property
    def dataset(self):
        return self.df

    def process_dataset(self):
        for cell_type, processor in self.mapping:
            md_mask = (self.df['cell_type'] == cell_type)
            self.df['processed_source'] = None
            self.df.loc[md_mask, 'processed_source'] = self.df[md_mask].source.apply(
                lambda row: processor.process(row)
            )
