# MolErr2Fix

A comprehensive benchmark dataset for evaluating Large Language Models' capabilities in detecting, localizing, explaining, and correcting chemical errors in molecular descriptions.

Paper: https://huggingface.co/papers/2509.00063 (**EMNLP 2025 Oral**)

Dataset: https://huggingface.co/datasets/YoungerWu/MolErr2Fix


## 🧬 Overview

MolErr2Fix is a specialized benchmark designed to assess how well Large Language Models (LLMs) can handle molecular chemistry error analysis tasks. The benchmark evaluates four core capabilities:

1. **Error Detection**: Identifying whether errors exist in molecular descriptions
2. **Error Localization**: Pinpointing the exact location of errors within text
3. **Error Reasoning**: Explaining why the identified segments are incorrect
4. **Error Revision**: Providing corrected versions of erroneous descriptions

## 📊 Dataset

The benchmark includes molecular structures in SMILES format paired with potentially erroneous textual descriptions. Each sample contains:

- **CID**: Compound identifier
- **SMILES**: Molecular structure representation
- **Initial Description**: Text description (potentially containing errors)
- **Error Classifications**: Categorized error types
- **Error Locations**: Specific text segments with errors
- **Ground Truth**: Correct descriptions

### Error Types

The benchmark defines 6 main categories of molecular description errors:

- **E1 - Functional Group/Substituent Error**: Misidentification of molecular substructures
- **E2 - Classification Error**: Incorrect molecular classification
- **E3 - Derivation Error**: Wrong derivation or origin descriptions
- **E4 - Stereochemistry Error**: Incorrect stereochemical information
- **E5 - Sequence/Composition Error**: Wrong sequence or compositional details
- **E6 - Indexing Error**: Incorrect numbering or indexing

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/HeinzVonHank/MolErr2Fix.git
cd MolErr2Fix
pip install -r requirements.txt
```

### Environment Setup

Set up your API keys:
```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export XAI_API_KEY="your_xai_key"  # For Grok
```

### How to Run

#### 1. Run Tests (No API Key Needed)
```bash
python run_tests.py
```
This will verify that all components work correctly.

#### 2. Error Detection
```bash
# Evaluate all samples
python eval.py detection --model gpt-4o --start 0 --end -1

# Evaluate first 10 samples
python eval.py detection --model gpt-4o --start 0 --end 10

# Using shell script
bash scripts/run_detection.sh gpt-4o 0 10
```

#### 3. Error Localization
```bash
# Evaluate first 10 samples
python eval.py localization --model gpt-4o --start 0 --end 10

# Using shell script
bash scripts/run_localization.sh gpt-4o 0 10
```

#### 4. Error Reasoning
```bash
# Evaluate first 5 samples
python eval.py reasoning --model gpt-4o --eval_model gpt-4o --start 0 --end 5

# Using shell script
bash scripts/run_reasoning.sh gpt-4o gpt-4o 0 5
```

#### 5. Error Revision
```bash
# Evaluate first 5 samples
python eval.py revision --model gpt-4o --eval_model gpt-4o --start 0 --end 5

# Using shell script
bash scripts/run_revision.sh gpt-4o gpt-4o 0 5
```

**Note:** Results will be saved to `results/<model_name>/` directory.

## 📁 Project Structure

```
MolErr2Fix/
├── src/
│   ├── evaluators/
│   │   ├── error_detector.py      # Error detection evaluator
│   │   ├── error_localization.py  # Error localization evaluator
│   │   ├── error_reasoning.py     # Error reasoning evaluator
│   │   └── error_revision.py      # Error revision evaluator
│   ├── dataloader.py              # Dataset loading utilities
│   ├── llm_interface.py           # LLM API interfaces
│   └── utils.py                   # Helper functions
├── configs/
│   └── error_types.yaml           # Error type definitions
├── scripts/
│   ├── run_detection.sh           # Detection evaluation script
│   ├── run_localization.sh        # Localization evaluation script
│   ├── run_reasoning.sh           # Reasoning evaluation script
│   └── run_revision.sh            # Revision evaluation script
├── eval.py                        # Main evaluation entry point
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🔧 Configuration

### Model Support

The benchmark supports multiple LLM providers:
- OpenAI GPT models (gpt-4, gpt-4o, gpt-4-turbo, etc.)
- Google Gemini models (gemini-2.0-flash, gemini-pro, etc.)
- Anthropic Claude models (claude-3-7-sonnet, claude-3-opus, etc.)
- xAI Grok models (grok-3, etc.)

### Command Line Arguments

Common arguments for `eval.py`:
- `task`: Evaluation task (detection, localization, reasoning, revision)
- `--model`: Model name for generation
- `--eval_model`: Model name for evaluation (reasoning/revision tasks)
- `--start`: Start index (default: 0)
- `--end`: End index (default: -1 for all samples)
- `--output_dir`: Output directory (default: results)
- `--verbose`: Enable verbose output

## 📈 Evaluation Metrics

The benchmark uses comprehensive metrics for evaluation:

- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: For localization tasks
- **Exact Match**: For exact text matching
- **BLEU Score**: For text generation quality

## 🔬 Research Applications

This benchmark is ideal for:

- Evaluating LLM performance on domain-specific chemistry tasks
- Developing better molecular description models
- Training specialized chemistry-aware language models
- Comparing different model architectures on chemical reasoning
- Building automated chemistry education tools

## 📊 Results Format

Evaluation results are saved in JSON format containing:

```json
{
  "overall": {
    "precision": 0.85,
    "recall": 0.78,
    "f1_score": 0.81
  },
  "per_error_type": {
    "E1": {"precision": 0.90, "recall": 0.85, "f1_score": 0.87},
    "E2": {"precision": 0.80, "recall": 0.75, "f1_score": 0.77}
  },
  "results": {
    "TP": ["E1", "E2"],
    "FP": ["E3"],
    "FN": ["E4"]
  }
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to:

- Report bugs or issues
- Suggest new error types or evaluation metrics
- Add support for new LLM providers
- Improve documentation
- Submit performance benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use MolErr2Fix in your research, please cite:

```bibtex
@article{wu2025molerr2fix,
  title={MolErr2Fix: Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision},
  author={Wu, Yuyang and Ye, Jinhui and Zhang, Shuhao and Dai, Lu and Bisk, Yonatan and Isayev, Olexandr},
  journal={arXiv preprint arXiv:2509.00063},
  year={2025}
}
```

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Keywords**: Molecular Chemistry, Large Language Models, Error Detection, Benchmark Dataset, Chemical Information Processing
