import re
import numpy as np
import pandas as pd
from datasets import load_dataset


ERROR_TO_CODE = {
    "Functional Group/Substituent Error": "E1",
    "Classification Error": "E2",
    "Derivation Error": "E3",
    "Stereochemistry Error": "E4",
    "Sequence/Composition Error": "E5",
    "Indexing Error": "E6"
}

CODE_TO_ERROR = {v: k for k, v in ERROR_TO_CODE.items()}

ERR_TYPE_BLOCK_RE = re.compile(r'\d\.\s*\[+.+Error\]+\:?')
ERR_TYPE_ONLY_RE = re.compile(r'\[+(.*?)\]+')
SENTENCE_INDEX_RE = re.compile(r'^\d\.\s+')


class MolBenchFormatter:
    """Formatter for MolErr2Fix benchmark dataset."""

    def __init__(self, dataset_name="YoungerWu/MolErr2Fix", error_dict=None, split='train'):
        self.dataset_name = dataset_name
        self.split = split
        self.error_dict = error_dict or {k.replace(' ', ''): v for k, v in ERROR_TO_CODE.items()}

        ds = load_dataset(self.dataset_name)
        self.db = ds[self.split].to_pandas()
        self._preprocess_dataframe()

    def _preprocess_dataframe(self):
        self.db['Parsed where wrong'] = self.db['Error_Spans'].apply(
            lambda x: [re.sub(SENTENCE_INDEX_RE, '', l) for l in x.strip().split('\n')]
        )
        self.db['Parsed Wrong Class/ Reasons'] = self.db['Error_Types_and_Reasons'].apply(self.parse_reason_block)
        self.db['Parsed Correct'] = self.db['Corrected_Description'].apply(
            lambda x: [re.sub(SENTENCE_INDEX_RE, '', l) for l in x.strip().split('\n')]
        )

        error_counts = self.db['Parsed where wrong'].apply(len)
        mismatch = np.where(error_counts != self.db['Parsed Wrong Class/ Reasons'].apply(len))[0]
        assert mismatch.shape[0] == 0, f'Length mismatch at index: {mismatch}'
        mismatch = np.where(error_counts != self.db['Parsed Correct'].apply(len))[0]
        assert mismatch.shape[0] == 0, f'Length mismatch at index: {mismatch}'

    def parse_reason_block(self, text):
        """Parse error type and reason from text block."""
        ret = []
        for l in text.split('\n'):
            if not l.strip():
                continue
            l = l.replace('\xa0', ' ')
            err_search = re.search(ERR_TYPE_BLOCK_RE, l)
            if not err_search:
                continue

            err = re.findall(ERR_TYPE_ONLY_RE, err_search.group(0))[0]
            err_code = self.error_dict.get(err.replace(' ', ''))

            if not err_code:
                continue

            explain = re.split(ERR_TYPE_BLOCK_RE, l)[-1].strip()
            ret.append((err_code, explain))
        return ret

    def load(self, start=0, end=-1, return_json=False):
        """Load dataset subset."""
        if end == -1:
            end = len(self.db)

        subset_db = self.db.iloc[start:end]

        if return_json:
            error_samples = []
            for idx in range(len(subset_db)):
                row = subset_db.iloc[idx]
                sample = row[['CID', 'SMILES', 'Ground_Truth_Description', 'Generated_Description']].to_dict()
                sample['index'] = str(row['CID'])
                sample['inside_errors'] = {}
                for j in range(len(row['Parsed where wrong'])):
                    err_tuple = row['Parsed Wrong Class/ Reasons'][j]
                    if len(err_tuple) == 2:
                        sample['inside_errors'][j + 1] = {
                            'wrong_point': row['Parsed where wrong'][j],
                            'wrong_type': err_tuple[0],
                            'wrong_reason': err_tuple[1],
                            'correct_point': row['Parsed Correct'][j]
                        }
                error_samples.append(sample)
            return error_samples
        else:
            return subset_db

    def copy(self):
        formatter = object.__new__(MolBenchFormatter)
        formatter.db = self.db.copy()
        formatter.error_dict = self.error_dict
        formatter.dataset_name = self.dataset_name
        formatter.split = self.split
        return formatter

    def dump(self, save_to):
        """Save dataframe to CSV."""
        if save_to:
            self.db.to_csv(save_to, index=False)


ALL_LOADED = MolBenchFormatter()
