import tqdm, json
from string import Template
from typing import List, Dict, Tuple


from dataloader import *
from llm_interface import get_llm_interfance

class ErrorCorrectionEvaluator:
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

    def evaluate_all(self, verbose=False) -> List[Dict]:
        N = self.molbench_formatter.db.shape[0]
        all_revised = []
        for idx in tqdm.tqdm(range(N)):
            row = self.molbench_formatter.db.iloc[idx]
            CID = row['CID']
            smiles = row['SMILES']
            description = row['Initial Description']
            block = []
            for wrong_segment in row['Parsed where wrong']:
                prompt = self.prompt_template.substitute(
                    smiles=smiles,
                    description=description,
                    wrong_segment=wrong_segment
                )

                try:
                    llm_response = self.interface.inference_text_only(
                        query=prompt,
                        system_message="You are a helpful assistant.",
                        temperature=0.7,
                        max_tokens=150
                    )
                    llm_response = llm_response.strip()
                    block.append(llm_response)

                    if verbose:
                        print(f"CID:\n{CID}\nPrompt:\n{prompt}\nllm response:\n{llm_response}")

                except Exception as e:
                    print(f"[Exception] Sample {CID} error {wrong_segment}: {str(e)}")
                    self.problems.append((smiles, description, wrong_segment))
                    block.append('')

            all_revised.append(block)

        ret = self.molbench_formatter.copy()
        ret.db['LLM Revised'] = all_revised
        ret.dump(self.save_to)
        return ret
    
class LLMRevisionResultEvaluator:
    def __init__(self, interface, molbench_formatter, save_to='result.json'):
        self.interface = interface
        if isinstance(molbench_formatter, str):
            self.molbench_formatter = MolBenchFormater(molbench_formatter)
        else:
            self.molbench_formatter = molbench_formatter

        # This shouldn't be necessary, need to check
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

    def evaluate_all(self, verbose=False) -> List[Dict]:
        N = self.molbench_formatter.db.shape[0]
        ret = {"total": 0, "match": 0}
        for idx in tqdm.tqdm(range(N)):
            row = self.molbench_formatter.db.iloc[idx]
            CID = row['CID']
            description = row['Initial Description']
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

                try:
                    llm_response = self.interface.inference_text_only(
                        query=prompt,
                        system_message="You are a strict and concise evaluator.",
                        temperature=0.0,
                        max_tokens=100
                    )
                    llm_response = int(llm_response.strip())
                    block.append(llm_response)

                    if verbose:
                        print(f"CID:\n{CID}\nPrompt:\n{prompt}\nllm response:\n{llm_response}")

                except Exception as e:
                    print(f"[Exception] Sample {CID} error {wrong_segment}: {str(e)}")
                    self.problems.append((CID, description, wrong_segment))

            ret['total'] += len(block)
            ret['match'] += sum(block)

        ret['accuracy'] = ret['match'] / ret['total']
        with open(self.save_to, 'w') as f:
            json.dump(ret, f, indent=4)
        return ret
    

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Error Revision test")
    parser.add_argument('--read_from', type=str, default="ALL", help="Path to dataset csv")
    parser.add_argument('--error_yaml', type=str, default="./error_type_new.yaml", help="Path to error definition YAML")
    parser.add_argument('--revise_model', type=str, default="claude-3-7-sonnet-20250219", help="Model name to generate revised text")
    parser.add_argument('--eval_model', type=str, default="gpt-4o", help="Model name to evaluate the revision")
    parser.add_argument('--index_range', type=str, default=None, help='index range in the form of "(start_index,end_index)"')
    args = parser.parse_args()

    if args.read_from == "ALL":
        loaded_data = ALL_LOADED
    else:
        loaded_data = MolBenchFormater(args.read_from)

    start, end = eval(args.index_range) if args.index_range is not None else (0, -1)
    #loaded_data.load(start, end)

    #llm_api = get_llm_interfance(args.revise_model)

    save_to = Path("./test_results") / args.revise_model

    #generator = ErrorCorrectionEvaluator(llm_api, loaded_data, save_to=(save_to/'revised_by_llm.csv'))
    #revised = generator.evaluate_all(verbose=False)

    revised = MolBenchFormater('/home/dominic/workspace/MOL-Reasoning/benchmark/test_results/claude-3-7-sonnet-20250219/revised_by_llm.csv')
    llm_api = get_llm_interfance(args.eval_model)
    evaluator = LLMRevisionResultEvaluator(llm_api, revised, save_to=(save_to/'revision_performance_eval_by_gpt-4o.json'))
    result = evaluator.evaluate_all(verbose=True)