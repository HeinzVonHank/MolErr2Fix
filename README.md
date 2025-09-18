# MolErr2Fix

A comprehensive benchmark dataset for evaluating Large Language Models' capabilities in detecting, localizing, explaining, and correcting chemical errors in molecular descriptions.

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
git clone https://github.com/yourusername/MolErr2Fix.git
cd MolErr2Fix
pip install -r requirements.txt
```

### Basic Usage

#### Error Detection
```bash
python benchmark/ErrorDetector.py --model gemini-2.0-flash --read_from ALL
```

#### Error Localization
```bash
python benchmark/ErrorLocalization.py --model gpt-4 --error_yaml ./benchmark/error_type_new.yaml
```

#### Error Reasoning
```bash
python benchmark/ErrorReasoning.py --model claude-3-sonnet --dataset_path ./data/
```

#### Error Revision
```bash
python benchmark/ErrorRevision.py --model gpt-4-turbo --save_to ./results/revised.csv
```

## 📁 Project Structure

```
MolErr2Fix/
├── benchmark/
│   ├── ErrorDetector.py      # Error detection evaluation
│   ├── ErrorLocalization.py  # Error localization assessment
│   ├── ErrorReasoning.py     # Error explanation evaluation
│   ├── ErrorRevision.py      # Error correction assessment
│   ├── dataloader.py         # Dataset loading utilities
│   ├── llm_interface.py      # LLM API interfaces
│   ├── utilities.py          # Helper functions
│   └── error_type_new.yaml   # Error type definitions
├── LICENSE                   # MIT License
└── README.md                # This file
```

## 🔧 Configuration

### Model Support

The benchmark supports multiple LLM providers:
- OpenAI GPT models (GPT-3.5, GPT-4, GPT-4-turbo)
- Google Gemini models
- Anthropic Claude models
- Custom model interfaces

### Environment Variables

Set up your API keys:
```bash
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_google_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

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
