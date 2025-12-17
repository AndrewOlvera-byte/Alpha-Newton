<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.7+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.40+-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/TRL-0.8+-blue" alt="TRL">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs Welcome">
</p>

<h1 align="center">Alpha-Newton</h1>

<p align="center">
  <strong>A modular LLM post-training framework for building instruction-tuned and reasoning-capable language models.</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Features

- **Modular Registry System** — Clean separation of concerns with pluggable components for models, tokenizers, datasets, and trainers
- **Multi-Stage Training Pipeline** — SFT → DPO workflow for building aligned language models
- **Mixed Dataset Training** — Combine multiple datasets with weighted sampling for diverse training
- **Flash Attention 2** — Optimized attention for faster training and lower memory usage
- **Sequence Packing** — Efficient batching by concatenating sequences to maximize GPU utilization
- **WandB Integration** — Real-time training metrics and experiment tracking
- **Gradio Chat Interface** — Interactive web UI to test your trained models
- **Docker-First** — Reproducible environments with GPU support out of the box

---

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/Alpha-Newton.git
cd Alpha-Newton

# Build and start the container
docker compose build
docker compose up -d

# Verify GPU access
docker compose exec alpha-newton nvidia-smi
```

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

### Requirements

- Python 3.10+
- CUDA 12.0+ with compatible GPU (16GB+ VRAM recommended)
- Docker & Docker Compose (for containerized setup)

---

## Quick Start

### 1. Train an Instruction Model (SFT)

```bash
# Using Docker
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp qwen3_600M_sft_mixed_optimized

# Or locally
python src/entrypoints/train_sft.py --exp qwen3_600M_sft_mixed_optimized
```

This trains Qwen3-0.6B on a mix of OpenHermes-2.5 (70%) and Glaive function-calling (30%) datasets.

### 2. Align with Human Preferences (DPO)

```bash
docker compose exec alpha-newton python src/entrypoints/train_dpo.py \
  --exp qwen3_600M_dpo_anthropic_hh
```

DPO (Direct Preference Optimization) fine-tunes the model on preference pairs from Anthropic HH-RLHF.

### 3. Evaluate on Math Benchmarks (GSM8K, MATH)

This project includes a lightweight evaluation entrypoint that runs ecosystem-standard benchmarks via **EleutherAI lm-evaluation-harness**.

Install eval-only dependencies (kept separate from training deps):

```bash
pip install -r requirements/eval.txt
```

Run a math suite using the same `--exp` experiment naming pattern as training:

```bash
# Evaluate the checkpoint saved at training.output_dir
python src/entrypoints/eval_math.py --exp qwen3_600M_rlvr_math --suite math_small

# Full math suite (GSM8K + MATH)
python src/entrypoints/eval_math.py --exp qwen3_600M_rlvr_math --suite math_full

# Smoke test (limit samples)
python src/entrypoints/eval_math.py --exp qwen3_600M_rlvr_math --suite math_small --limit 100
```

Suites live in `configs/eval/*.yaml`, and can be swapped by changing the task list. By default we pass `--apply_chat_template`, which wraps each benchmark prompt into an OpenAI-style messages list (user role) and uses your tokenizer chat template for formatting.

### 3. Chat with Your Model

```bash
docker compose exec alpha-newton python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_dpo_anthropic_hh

# Or with a specific checkpoint
python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_sft_mixed_optimized \
  --checkpoint checkpoint-5000

# Create a public share link
python src/entrypoints/chat.py --model outputs/your_model --share
```

Access the web UI at **http://localhost:7860**

---

## Architecture

Alpha-Newton uses a **registry-based architecture** that makes it easy to extend and customize:

```
src/
├── core/
│   ├── config.py      # YAML config loading with variable interpolation
│   └── registry.py    # Component registry for models, data, trainers
├── builders/
│   ├── model.py       # Model builders (HuggingFace, custom)
│   ├── tokenizer.py   # Tokenizer builders
│   ├── data.py        # Dataset builders (SFT, DPO, mixed)
│   └── trainer.py     # Trainer builders (TRL SFT, DPO)
└── entrypoints/
    ├── train_sft.py   # SFT training script
    ├── train_dpo.py   # DPO training script
    └── chat.py        # Gradio chat interface
```

### How It Works

1. **Config Loading** — Merges `configs/base/common.yaml` with experiment-specific configs
2. **Component Building** — Registry system instantiates models, tokenizers, datasets, and trainers
3. **Training** — TRL-based trainers handle the training loop with WandB logging
4. **Inference** — Gradio interface loads checkpoints for interactive testing

---

## Configuration

### Config Structure

```yaml
# configs/exp/my_experiment.yaml

run:
  name: "my_experiment"      # Used for output directory and WandB
  mode: "sft"                # Training mode: sft, dpo

model:
  type: "hf_causal"          # Registry type
  id: "Qwen/Qwen3-0.6B-Base" # HuggingFace model ID
  dtype: "bfloat16"
  gradient_checkpointing: true

tokenizer:
  type: "hf"
  id: "Qwen/Qwen3-0.6B-Base"

data:
  type: "sft_mixed"          # Dataset type: sft, dpo, sft_mixed
  datasets_config:
    - path: "teknium/OpenHermes-2.5"
      weight: 0.7
    - path: "glaiveai/glaive-function-calling-v2"
      weight: 0.3
  max_seq_len: 2048
  packing: true              # Pack sequences for efficiency

training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  max_steps: 10000
  warmup_steps: 500
  save_steps: 1000

wandb:
  project: "posttraining"
  tags: ["qwen3", "sft", "mixed"]
```

### Variable Interpolation

Use `${section.key}` syntax for dynamic values:

```yaml
run:
  name: "my_experiment"

training:
  output_dir: "outputs/${run.name}"  # Resolves to "outputs/my_experiment"

wandb:
  run_name: "${run.name}"
```

### Available Dataset Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `sft` | Single dataset SFT | `train_path`, `eval_path` |
| `sft_mixed` | Multi-dataset with weights | `datasets_config` |
| `dpo` | Preference pairs | `train_path` (chosen/rejected) |

### Supported Dataset Formats

The framework auto-detects and converts these formats:

- **OpenAI Messages**: `{"messages": [{"role": "user", "content": "..."}]}`
- **OpenHermes**: `{"conversations": [{"from": "human", "value": "..."}]}`
- **Capybara**: `{"conversation": [{"input": "...", "output": "..."}]}`
- **Glaive**: `{"system": "...", "chat": "USER: ... ASSISTANT: ..."}`

---

## Training Pipeline

### Recommended Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Base Model    │────▶│   SFT Training  │────▶│  DPO Alignment  │
│  (Qwen, Llama)  │     │  (Instructions) │     │  (Preferences)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Chat Interface │
                                                │    (Gradio)     │
                                                └─────────────────┘
```

### Example Experiments

| Experiment | Description | Estimated Time |
|------------|-------------|----------------|
| `qwen3_600M_sft_mixed_optimized` | Mixed SFT (OpenHermes + Glaive) | ~3-4 hours |
| `qwen3_600M_dpo_anthropic_hh` | DPO on Anthropic HH-RLHF | ~30-45 min |
| `qwen3_600M_sft_tools` | Tool/function calling focus | ~2-3 hours |

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- **Bug Reports** — Found a bug? Open an issue with reproduction steps
- **Feature Requests** — Have an idea? We'd love to hear it
- **Code Contributions** — Submit a PR for bug fixes or new features
- **Documentation** — Help improve docs, tutorials, or examples
- **New Datasets** — Add support for additional dataset formats
- **New Training Methods** — Extend the training pipeline with additional algorithms

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/your-username/Alpha-Newton.git
cd Alpha-Newton

# Create a branch for your feature
git checkout -b feature/my-awesome-feature

# Install in development mode
pip install -e .

# Make your changes and test
python test_pipeline.py

# Submit a PR!
```

### Adding a New Component

The registry system makes it easy to add new components:

```python
# src/builders/my_component.py
from src.core.registry import register

@register("data", "my_custom_dataset")
def build_my_dataset(tokenizer, **kwargs):
    # Your dataset loading logic
    return {"train": train_dataset, "eval": eval_dataset}
```

Then use it in your config:

```yaml
data:
  type: "my_custom_dataset"
  # your custom parameters
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to public functions
- Keep functions focused and modular

### Pull Request Guidelines

1. **Create an issue first** for significant changes
2. **Write clear commit messages** describing what and why
3. **Include tests** if adding new functionality
4. **Update documentation** if changing behavior
5. **Keep PRs focused** — one feature/fix per PR

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with these amazing libraries:

- [Transformers](https://github.com/huggingface/transformers) — Model loading and inference
- [TRL](https://github.com/huggingface/trl) — Training algorithms (SFT, DPO)
- [Datasets](https://github.com/huggingface/datasets) — Efficient data loading
- [WandB](https://wandb.ai) — Experiment tracking
- [Gradio](https://gradio.app) — Web interface

