"""
Quick test script to verify evaluation pipeline without API keys.
"""
import sys
sys.path.insert(0, '.')

from src.dataloader import ALL_LOADED
from src.utils import load_yaml
from src.evaluators.error_detector import ErrorDetector


class MockLLMInterface:
    """Mock LLM interface for testing without API keys."""

    def inference_text_only(self, query, system_message='', temperature=0.7, max_tokens=1024):
        return '["E1", "E2"]'


def test_detection():
    print("Testing Error Detection pipeline...")

    error_dict = load_yaml('configs/error_types.yaml')
    molbench = ALL_LOADED.copy()
    molbench.db = molbench.db.iloc[0:2]

    mock_api = MockLLMInterface()
    evaluator = ErrorDetector(mock_api, molbench, error_dict)

    print(f"  Dataset size: {len(molbench.db)}")
    print(f"  Error types: {len(error_dict)}")

    results = evaluator.evaluate_all(verbose=False)
    metrics = evaluator.compute_metrics(results)

    print(f"  TP: {len(results['TP'])}")
    print(f"  FP: {len(results['FP'])}")
    print(f"  FN: {len(results['FN'])}")
    print(f"  Overall F1: {metrics['overall']['f1_score']:.3f}")
    print("✓ Detection pipeline works!")


def test_data_loading():
    print("\nTesting data loading with subsets...")

    molbench = ALL_LOADED.copy()
    print(f"  Full dataset: {len(molbench.db)} samples")

    molbench.db = molbench.db.iloc[0:10]
    print(f"  Subset [0:10]: {len(molbench.db)} samples")

    samples = molbench.load(start=0, end=-1, return_json=True)
    print(f"  JSON format: {len(samples)} samples")
    print(f"  Sample keys: {list(samples[0].keys())}")
    print("✓ Data loading works!")


if __name__ == "__main__":
    print("=" * 60)
    print("MolErr2Fix Evaluation Pipeline Test")
    print("=" * 60)

    test_data_loading()
    test_detection()

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
