import io
import re
import tokenize

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
            # normal fix for zero code cells
            'normalized_defined_functions': lambda x: (len(self._get_defined_functions(x))
                                                       / self.group_params['code_count']),
            'normalized_sloc': self._get_normalized_sloc,
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
        total_md = types_count.markdown if 'markdown' in types_count else 0
        total_code = types_count.code if 'code' in types_count else 1
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
    def _get_functions_by_code(source):
        try:
            code_text = tokenize.generate_tokens(io.StringIO(source).readline)
            token_strings = [tok.string for tok in code_text if tok.type == 1]
        except (tokenize.TokenError, IndentationError):
            return [], []

        return token_strings
