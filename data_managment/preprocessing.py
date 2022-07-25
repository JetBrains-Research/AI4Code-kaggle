import re

import nltk
import pandas as pd
import tldextract
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
        return lambda s: clean_text(" ".join([i.text for i in s.find_all(search_fun)]))

    def process(self, md_string):
        soup = BeautifulSoup(markdown(md_string), "html.parser")
        res = {}

        for name, fun in self.features.items():
            res[name] = fun(soup)

        return res


def preprocess_dataframe(df: DataFrame) -> DataFrame:
    md_processor = MdProcessor()
    df_md = df.groupby('cell_type').get_group('markdown').set_index(['id', 'cell_id'])
    processed_data_df = df_md.source.apply(lambda x: pd.Series(md_processor.process(x)))

    return df.merge(processed_data_df, on=['id', 'cell_id'], how='left')


class ImprovedMDPProcessor:

    def __init__(self):
        self.tag_dict = {
            'a': '@',
            'b': '**',
            'code': "'''",
            'em': '!!',
            'h1': '-',
            'h2': '--',
            'h3': '---',
            'h4': '----',
            'h5': '-----',
            'h6': '------',
            'hr': '_-_',
            'i': '__',
            'strong': '!-',
            'title': '!-!',
            'font': '^^',
            'img': '§',
            'ol': '№',
            'ul': '$',
            'pre': '±',
            'section': '<->',
            'span': '<>', }

    def _process_attrs(self, soup):
        for tag in soup.findAll():
            if tag.name not in self.tag_dict.keys():
                tag.unwrap()
            elif len(tag.attrs) > 0:
                if 'href' in tag.attrs:
                    tag = self._process_links(tag)
                if 'alt' in tag.attrs:
                    tag = self._process_img(tag)

                tag.attrs = {}

        return soup

    @staticmethod
    def _process_links(tag):
        link = tag.attrs['href']
        link = tldextract.extract(link)

        tag.string = tag.text + ' ' + link[1] + ' ' + link[2]

        return tag

    @staticmethod
    def _process_img(tag):
        link = tag.attrs['alt']
        tag.string = tag.text + ' ' + link

        return tag

    def process(self, md_string):
        soup = BeautifulSoup(markdown(md_string), "html.parser")
        soup = self._process_attrs(soup)

        for node in soup.find_all(text=lambda x: x.strip()):
            node.replace_with(kaggle_cleaning(node))

        md_string = str(soup)

        pattern = r''
        for tag, new_tag in self.tag_dict.items():
            md_string = md_string.replace(f'<{tag}>', f' {new_tag} ')
            pattern += f'</{tag}>|'
            pattern += f'<{tag}/>|'

        md_string = re.sub(pattern[:-1], "", md_string)
        md_string = re.sub(r"\s+", " ", md_string, flags=re.I)
        md_string = md_string.strip()

        return md_string


class DatasetProcessor:
    def __init__(self, path, processor=ImprovedMDPProcessor):
        self.df = pd.read_feather(path) if isinstance(path, str) else path
        self.mapping = {"markdown": processor()}

    @property
    def dataset(self):
        return self.df

    def process_dataset(self):
        for cell_type, processor in self.mapping.items():
            cell_mask = self.df["cell_type"] == cell_type
            self.df = self.df.assign(processed_source='')
            # self.df.loc[:, ""] = None
            self.df.loc[cell_mask, "processed_source"] = self.df[cell_mask].source.progress_apply(
                lambda row: processor.process(row)
            )
        return self.df
