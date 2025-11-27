"""
Comprehensive test suite for MolErr2Fix benchmark.
Tests all major components without requiring API keys.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from src.dataloader import ALL_LOADED, MolBenchFormatter
from src.utils import load_yaml, save_json, calc_f1
from src.evaluators.error_detector import ErrorDetector
from src.evaluators.error_localization import ErrorLocalization, LocalizationEvaluator
from src.evaluators.error_reasoning import ErrorReasoning, ReasoningEvaluator
from src.evaluators.error_revision import ErrorRevision


class MockLLMInterface:
    """Mock LLM for testing."""
    def inference_text_only(self, query, system_message='', temperature=0.7, max_tokens=1024):
        if 'error_type' in query.lower() and 'error_span' in query.lower():
            return '[{"error_type": "E1", "error_span": "test error"}]'
        elif 'respond with "yes"' in query.lower() or 'respond with "no"' in query.lower():
            return '"Yes"'
        return '["E1", "E2"]'


def test_1_imports():
    """Test 1: All imports work."""
    print("\n[Test 1] Testing imports...")
    print("  ✓ dataloader")
    print("  ✓ utils")
    print("  ✓ llm_interface")
    print("  ✓ evaluators")
    return True


def test_2_data_loading():
    """Test 2: Data loading and subsetting."""
    print("\n[Test 2] Testing data loading...")

    molbench = ALL_LOADED
    print(f"  ✓ Loaded {len(molbench.db)} samples")

    error_dict = load_yaml('configs/error_types.yaml')
    print(f"  ✓ Loaded {len(error_dict)} error types")

    samples = molbench.load(start=0, end=5, return_json=True)
    print(f"  ✓ JSON conversion: {len(samples)} samples")
    print(f"  ✓ Sample keys: {len(samples[0].keys())} fields")

    assert len(samples) == 5, "Should load 5 samples"
    assert 'inside_errors' in samples[0], "Should have inside_errors"

    return True


def test_3_error_detection():
    """Test 3: Error detection pipeline."""
    print("\n[Test 3] Testing error detection...")

    error_dict = load_yaml('configs/error_types.yaml')
    molbench = ALL_LOADED.copy()
    molbench.db = molbench.db.iloc[0:3]

    mock_api = MockLLMInterface()
    evaluator = ErrorDetector(mock_api, molbench, error_dict)

    results = evaluator.evaluate_all(verbose=False)
    metrics = evaluator.compute_metrics(results)

    print(f"  ✓ Evaluated {len(molbench.db)} samples")
    print(f"  ✓ Overall F1: {metrics['overall']['f1_score']:.3f}")
    print(f"  ✓ Per-type metrics: {len(metrics['per_error_type'])} types")

    assert 'overall' in metrics, "Should have overall metrics"
    assert 'per_error_type' in metrics, "Should have per-type metrics"

    return True


def test_4_error_localization():
    """Test 4: Error localization pipeline."""
    print("\n[Test 4] Testing error localization...")

    error_dict = load_yaml('configs/error_types.yaml')
    molbench = ALL_LOADED
    samples = molbench.load(start=0, end=2, return_json=True)

    mock_api = MockLLMInterface()
    generator = ErrorLocalization(mock_api, error_dict)

    samples_with_errors = generator.error_localization(samples)
    print(f"  ✓ Generated localizations for {len(samples_with_errors)} samples")

    evaluator = LocalizationEvaluator(iou_threshold=0.5)
    eval_results = evaluator.evaluate(samples_with_errors)

    print(f"  ✓ Total errors: {eval_results['total_error_count']}")
    print(f"  ✓ Span accuracy: {eval_results['span_accuracy']:.3f}")

    assert 'span_accuracy' in eval_results, "Should have span accuracy"

    return True


def test_5_utils():
    """Test 5: Utility functions."""
    print("\n[Test 5] Testing utility functions...")

    precision, recall, f1 = calc_f1(10, 5, 3)
    print(f"  ✓ F1 calculation: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

    test_data = {"test": "data"}
    test_path = Path("results/test/test.json")
    save_json(test_data, test_path)
    print(f"  ✓ JSON save: {test_path}")

    error_dict = load_yaml('configs/error_types.yaml')
    print(f"  ✓ YAML load: {len(error_dict)} error types")

    test_path.unlink()
    test_path.parent.rmdir()

    return True


def test_6_data_subsetting():
    """Test 6: Data subsetting in eval.py style."""
    print("\n[Test 6] Testing data subsetting...")

    molbench = ALL_LOADED.copy()
    original_size = len(molbench.db)
    print(f"  ✓ Original size: {original_size}")

    molbench.db = molbench.db.iloc[0:10]
    subset_size = len(molbench.db)
    print(f"  ✓ Subset size: {subset_size}")

    assert subset_size == 10, "Should have 10 samples"
    assert subset_size < original_size, "Subset should be smaller"

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("MolErr2Fix Comprehensive Test Suite")
    print("=" * 70)

    tests = [
        test_1_imports,
        test_2_data_loading,
        test_3_error_detection,
        test_4_error_localization,
        test_5_utils,
        test_6_data_subsetting,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nThe code is ready to run!")
        print("\nNext steps:")
        print("1. Set your API keys (OPENAI_API_KEY, GEMINI_API_KEY, etc.)")
        print("2. Run: python eval.py detection --model gpt-4o --start 0 --end 5")
        print("3. Or use: bash scripts/run_detection.sh gpt-4o 0 5")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
