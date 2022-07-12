import tokenize
import io
import re

import pandas as pd

plot_functions = {
    'histplot', 'kdeplot', 'displot',
    'displot', 'scatterplot', 'relplot',
    'jointplot', 'plot', 'show',
    'pairplot', 'countplot', 'jointplot',
    'relplot', 'lmplot', 'catplot',
    'scatter', 'hist',
}


class FeaturesProcessor:
    def __init__(self):
        self.group_params = {'md_count': -1, 'code_count': -1, 'source_code': ''}

        self.mapping = {
            'md_count': lambda x: self.group_params['md_count'],
            'code_count': lambda x: self.group_params['code_count'],
            'defined_functions': self._get_defined_functions,
            'normalized_plot_functions': self._get_normalized_plot_functions,
            'normalized_defined_functions': lambda x: (len(self._get_defined_functions(x))
                                                       / self.group_params['code_count']),
            'normalized_sloc': self._get_normalized_sloc,
            'code_subsample': self._get_code_subsample,
        }

    def process(self, grouped_df, input_features):
        self.group_params = self._preprocess_group(grouped_df)

        feature_names = (set(self.mapping.keys()) & set(input_features))
        features = {feature_name: self.mapping[feature_name](grouped_df)
                    for feature_name in feature_names}

        return pd.Series(features)

    @staticmethod
    def _preprocess_group(group) -> dict:
        types_count = group.cell_type.value_counts()
        total_md = types_count.markdown
        total_code = types_count.code
        code_sub_df = group[group.cell_type == "code"]
        source_code = '\n'.join(code_sub_df.source)

        return {'md_count': total_md, 'code_count': total_code, 'source_code': source_code}

    def _get_normalized_sloc(self, group) -> float:
        return len(self.group_params['source_code'].splitlines()) / self.group_params['code_count']

    @staticmethod
    def _get_defined_functions(group) -> str:
        code_sub_df = group[group.cell_type == "code"]
        source_code = '\n'.join(code_sub_df.source)

        functions = re.findall('def (.*)\(', source_code)
        return " ".join(functions)

    def _get_normalized_plot_functions(self, group) -> float:
        code_sub_df = group[group.cell_type == "code"]
        source_code = '\n'.join(code_sub_df.source)

        functions = self._get_functions_by_code(source_code)
        return len(functions) / self.group_params['code_count']

    @staticmethod
    def _get_code_subsample(group, n=20, seed=42):
        code_df = group[group.cell_type == "code"]

        n = len(code_df) if n > len(code_df) else n
        return ' '.join(code_df.source.sample(n, random_state=seed).astype(str).tolist())

    @staticmethod
    def _get_functions_by_code(source):
        try:
            code_text = tokenize.generate_tokens(io.StringIO(source).readline)
            token_strings = [tok.string for tok in code_text if tok.type == 1]
        except (tokenize.TokenError, IndentationError):
            return [], []

        return token_strings
