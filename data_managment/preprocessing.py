import re
import json

import pandas as pd
from pandas import DataFrame
from bs4 import BeautifulSoup
from markdown import markdown

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')

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
            "links_labels": self.rule2text("a"),
            "image_labels": self.get_image_labels,
            "links": self.get_links,
        }

    @staticmethod
    def rule2text(search_fun):
        return lambda s: [i.text for i in s.find_all(search_fun)]

    @staticmethod
    def get_image_labels(s):
        return [i['alt'] for i in s.find_all('img', alt=True)]

    @staticmethod
    def get_links(s):
        return [
            i['href'] if i.name == 'a' else i['src']
            for i in s.find_all('a') + s.find_all('img')
        ]

    def process(self, md_string):
        soup = BeautifulSoup(markdown(md_string), "html.parser")
        res = {}
        for name, fun in self.features.items():
            res[name] = fun(soup)

        return json.dumps(res)


def preprocess_dataframe(df: DataFrame) -> DataFrame:
    md_processor = MdProcessor()
    md_mask = df["cell_type"] == "markdown"

    df["processed_source"] = None
    df.loc[md_mask, "processed_source"] = df[md_mask].source.apply(
        lambda row: md_processor.process(row)
    )

    return df


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


if __name__ == '__main__':
    prc = MdProcessor()
    md = '![CNN Architecture](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)'
    md += '\nMy favorite search engine is [Duck Duck Go](https://duckduckgo.com).'
    print(prc.process(md))
