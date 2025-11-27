from string import Template
from tqdm import tqdm

from src.dataloader import MolBenchFormatter


class ErrorRevision:
    """Generate error corrections."""

    def __init__(self, interface, molbench_formatter, save_to='./revised_by_llm.csv'):
        self.interface = interface
        self.molbench_formatter = molbench_formatter.copy()
        self.problems = []
        self.save_to = save_to

        self.prompt_template = Template("""
            You are an expert in molecular structure and error correction.
            The molecule and its description below contain an error, and we have pointed out the erroneous segment for you.
            Molecule's structure (SMILES): $smiles
            Erroneous Description: $description
            Erroneous segment: " $wrong_segment "
            Please correct the error by providing a corrected substitution for the pointed out error segment,
            your answer should have a similar length as the erroneous segment,
            return the corrected segment only, and do not include any other text.
        """)

    def evaluate_all(self, verbose=False):
        """Generate corrections for all error segments."""
        N = self.molbench_formatter.db.shape[0]
        all_revised = []

        for idx in tqdm(range(N)):
            row = self.molbench_formatter.db.iloc[idx]
            CID = row['CID']
            smiles = row['SMILES']
            description = row['Generated_Description']
            block = []
            for wrong_segment in row['Parsed where wrong']:
                prompt = self.prompt_template.substitute(
                    smiles=smiles,
                    description=description,
                    wrong_segment=wrong_segment
                )

                llm_response = self.interface.inference_text_only(
                    query=prompt,
                    system_message="You are a helpful assistant.",
                    temperature=0.7,
                    max_tokens=150
                )
                llm_response = llm_response.strip()
                block.append(llm_response)

            all_revised.append(block)

        ret = self.molbench_formatter.copy()
        ret.db['LLM Revised'] = all_revised
        ret.dump(self.save_to)
        return ret


class RevisionEvaluator:
    """Evaluate LLM-generated revisions."""

    def __init__(self, interface, molbench_formatter, save_to='result.json'):
        self.interface = interface
        if isinstance(molbench_formatter, str):
            self.molbench_formatter = MolBenchFormatter(molbench_formatter)
        else:
            self.molbench_formatter = molbench_formatter

        if isinstance(self.molbench_formatter.db['LLM Revised'].iloc[0], str):
            self.molbench_formatter.db['LLM Revised'] = self.molbench_formatter.db['LLM Revised'].apply(eval)

        self.save_to = save_to
        self.problems = []

        self.prompt_template = Template("""
            You are an expert in molecular structure and error correction evaluation.
            Below is the original erroneous description:
            "$description"

            An error has been identified in the description as: "$wrong_segment"
            The human-corrected version for this error is: "$human_correct"
            The model-generated correction for this error is: "$llm_correct"

            Please evaluate whether the model-generated correction properly fixes the error by replacing the wrong segment with a fragment that conveys the same meaning as the human correction, and without modifying parts of the description that were not marked as errors.
            If the correction is appropriate, reply with 1 (and nothing else). Otherwise, reply with 0 (and nothing else).
        """)

    def evaluate_all(self, verbose=False):
        """Evaluate all LLM revisions."""
        N = self.molbench_formatter.db.shape[0]
        ret = {"total": 0, "match": 0}

        for idx in tqdm(range(N)):
            row = self.molbench_formatter.db.iloc[idx]
            CID = row['CID']
            description = row['Generated_Description']
            block = []
            for wrong_segment, human_correct, llm_correct in zip(row['Parsed where wrong'], row['Parsed Correct'], row['LLM Revised']):
                if not all([wrong_segment, human_correct, llm_correct]):
                    continue
                prompt = self.prompt_template.substitute(
                    description=description,
                    wrong_segment=wrong_segment,
                    human_correct=human_correct,
                    llm_correct=llm_correct
                )

                llm_response = self.interface.inference_text_only(
                    query=prompt,
                    system_message="You are a strict and concise evaluator.",
                    temperature=0.0,
                    max_tokens=100
                )
                llm_response = int(llm_response.strip())
                block.append(llm_response)

            ret['total'] += len(block)
            ret['match'] += sum(block)

        ret['accuracy'] = ret['match'] / ret['total'] if ret['total'] > 0 else 0
        return ret
