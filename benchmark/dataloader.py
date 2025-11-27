import numpy as np
import pandas as pd
import re
from typing import Dict, List
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# For anonymous access, no login is needed for public datasets.

ERROR_TO_CODE = {"Functional Group/Substituent Error": "E1", "Classification Error": "E2",
                 "Derivation Error": "E3", "Stereochemistry Error": "E4",
                 "Sequence/Composition Error": "E5", "Indexing Error": "E6"}

CODE_TO_ERROR = {v: k for k, v in ERROR_TO_CODE.items()}

ErrTypeBlockRe = re.compile(r'\d\.\s*\[+.+Error\]+\:?')
ErrTypeOnlyRe = re.compile(r'\[+(.*?)\]+')
SentenceIndexRe = re.compile(r'^\d\.\s+')

class MolBenchFormater:
    def __init__(self, dataset_name: str, error_dict: Dict = ERROR_TO_CODE, split: str = 'train'):
        self.dataset_name = dataset_name
        self.split = split
        self.error_dict = {k.replace(' ', ''): v for k, v in error_dict.items()}

        print(f"Loading dataset '{self.dataset_name}' with split '{self.split}' from Hugging Face...")
        try:
            ds = load_dataset(self.dataset_name)
            self.db = ds[self.split].to_pandas()
            self._preprocess_dataframe()
            print("Dataset loaded and preprocessed successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}")

    def _preprocess_dataframe(self):
        self.db['Parsed where wrong'] = self.db['Error_Spans'].apply(lambda x: [re.sub(SentenceIndexRe, '', l) for l in x.strip().split('\n')])
        self.db['Parsed Wrong Class/ Reasons'] = self.db['Error_Types_and_Reasons'].apply(self.parse_reason_block)
        self.db['Parsed Correct'] = self.db['Corrected_Description'].apply(lambda x: [re.sub(SentenceIndexRe, '', l) for l in x.strip().split('\n')])

        error_counts = self.db['Parsed where wrong'].apply(len)
        mismatch = np.where(error_counts != self.db['Parsed Wrong Class/ Reasons'].apply(len))[0]
        assert mismatch.shape[0] == 0, f'Parsed where wrong and Parsed Wrong Class/ Reasons length mismatch at index: {mismatch}'
        mismatch = np.where(error_counts != self.db['Parsed Correct'].apply(len))[0]
        assert mismatch.shape[0] == 0, f'Parsed where wrong and Parsed Correct length mismatch at index: {mismatch}'

    def parse_reason_block(self, text: str, verbose: bool = False) -> List[tuple]:
        ret = []
        for l in text.split('\n'):
            if not l.strip():
                continue
            l = l.replace('\xa0', ' ')
            err_search = re.search(ErrTypeBlockRe, l)
            if not err_search:
                if verbose:
                    print(f"Warning: No error block found in line: {l}")
                continue

            err = re.findall(ErrTypeOnlyRe, err_search.group(0))[0]
            err_code = self.error_dict.get(err.replace(' ', ''))
            
            if not err_code:
                if verbose:
                    print(f"Warning: Unknown error type '{err}' in line: {l}")
                continue

            explain = re.split(ErrTypeBlockRe, l)[-1].strip()
            ret.append((err_code, explain))
        return ret
    
    def load(self, start: int = 0, end: int = -1, return_json: bool = False):
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
        return MolBenchFormater(self.db.copy(), self.error_dict)
    
    def dump(self, save_to=None):
        if save_to:
            self.db.to_csv(save_to, index=False)
        else:
            print("No save_to path provided. Nothing dumped.")

ALL_LOADED = MolBenchFormater(dataset_name="YoungerWu/MolErr2Fix")