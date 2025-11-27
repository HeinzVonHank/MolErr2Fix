import argparse
import json
from pathlib import Path

from src.dataloader import ALL_LOADED
from src.llm_interface import get_llm_interface
from src.utils import load_yaml, save_json
from src.evaluators import ErrorDetector, ErrorLocalization, ErrorReasoning, ErrorRevision


def eval_detection(args):
    """Evaluate error detection task."""
    error_dict = load_yaml(args.error_yaml)
    molbench = ALL_LOADED.copy()

    if args.end != -1:
        molbench.db = molbench.db.iloc[args.start:args.end]
    elif args.start > 0:
        molbench.db = molbench.db.iloc[args.start:]

    llm_api = get_llm_interface(args.model)
    evaluator = ErrorDetector(llm_api, molbench, error_dict)

    results = evaluator.evaluate_all(verbose=args.verbose)
    metrics = evaluator.compute_metrics(results)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics["results"] = results
    save_json(metrics, output_dir / "detection_results.json")
    save_json(evaluator.problems, output_dir / "detection_problems.json")

    print(json.dumps(metrics, indent=4, ensure_ascii=False))


def eval_localization(args):
    """Evaluate error localization task."""
    error_dict = load_yaml(args.error_yaml)
    molbench = ALL_LOADED
    samples = molbench.load(start=args.start, end=args.end, return_json=True)

    llm = get_llm_interface(args.model)
    generator = ErrorLocalization(llm, error_dict)
    samples_with_errors = generator.error_localization(samples)

    output_dir = Path(args.output_dir) / "Localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(samples_with_errors, output_dir / f"localization_{args.model}_{args.end}.json")

    from src.evaluators.error_localization import LocalizationEvaluator
    evaluator = LocalizationEvaluator(iou_threshold=0.5)
    eval_results = evaluator.evaluate(samples_with_errors)
    save_json(eval_results, output_dir / f"evaluation_localization_{args.model}_{args.end}.json")

    print(f"\n=== Error Localization Evaluation ===")
    print(f"Total errors: {eval_results['total_error_count']}")
    print(f"Span accuracy (IoU ≥ 0.5): {eval_results['span_accuracy']:.3f}")
    print(f"Exact match accuracy: {eval_results['exact_match_accuracy']:.3f}")
    print(f"F1 Score: {eval_results['f1_score']:.3f}")


def eval_reasoning(args):
    """Evaluate error reasoning task."""
    error_dict = load_yaml(args.error_yaml)
    molbench = ALL_LOADED
    samples = molbench.load(start=args.start, end=args.end, return_json=True)

    gen_api = get_llm_interface(args.model)
    from src.evaluators.error_reasoning import ErrorReasoning, ReasoningEvaluator
    generator = ErrorReasoning(gen_api)
    samples_with_reasons = generator.generate_all_reasons(samples)

    output_dir = Path(args.output_dir) / "Reasoning"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(samples_with_reasons, output_dir / f"reasoning_{args.model}_{args.end}.json")

    eval_api = get_llm_interface(args.eval_model)
    evaluator = ReasoningEvaluator(eval_api)
    eval_results = evaluator.evaluate_stored_reasons(samples_with_reasons)
    save_json(eval_results, output_dir / f"evaluation_reasoning_{args.model}_by_{args.eval_model}_{args.end}.json")

    print("\n=== Reasoning Evaluation ===")
    print(f"Total: {eval_results['total_count']}")
    print(f"Matched: {eval_results['match_count']}")
    print(f"Accuracy: {eval_results['accuracy']:.3f}")


def eval_revision(args):
    """Evaluate error revision task."""
    molbench = ALL_LOADED.copy()

    if args.end != -1:
        molbench.db = molbench.db.iloc[args.start:args.end]
    elif args.start > 0:
        molbench.db = molbench.db.iloc[args.start:]

    llm_api = get_llm_interface(args.model)

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluators.error_revision import ErrorRevision, RevisionEvaluator
    generator = ErrorRevision(llm_api, molbench, save_to=str(output_dir / 'revised_by_llm.csv'))
    revised = generator.evaluate_all(verbose=args.verbose)

    eval_api = get_llm_interface(args.eval_model)
    evaluator = RevisionEvaluator(eval_api, revised, save_to=str(output_dir / f'revision_eval_by_{args.eval_model}.json'))
    result = evaluator.evaluate_all(verbose=args.verbose)

    print("\n=== Revision Evaluation ===")
    print(f"Total: {result['total']}")
    print(f"Matched: {result['match']}")
    print(f"Accuracy: {result['accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="MolErr2Fix Benchmark Evaluation")
    parser.add_argument('task', choices=['detection', 'localization', 'reasoning', 'revision'],
                        help="Evaluation task")
    parser.add_argument('--model', type=str, default="gpt-4o", help="Model name")
    parser.add_argument('--eval_model', type=str, default="gpt-4o", help="Evaluation model name")
    parser.add_argument('--error_yaml', type=str, default="configs/error_types.yaml", help="Error types config")
    parser.add_argument('--start', type=int, default=0, help="Start index")
    parser.add_argument('--end', type=int, default=-1, help="End index")
    parser.add_argument('--output_dir', type=str, default="results", help="Output directory")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    args = parser.parse_args()

    if args.task == 'detection':
        eval_detection(args)
    elif args.task == 'localization':
        eval_localization(args)
    elif args.task == 'reasoning':
        eval_reasoning(args)
    elif args.task == 'revision':
        eval_revision(args)


if __name__ == "__main__":
    main()
