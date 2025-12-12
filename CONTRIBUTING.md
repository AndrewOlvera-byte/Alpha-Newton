# Contributing to Alpha-Newton

First off, thank you for considering contributing to Alpha-Newton! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to democratize open language model training and push the capabilities of American open source AI. By participating, you are expected to:

- Be respectful and considerate in your communication
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 12.0+ with a compatible GPU
- Git
- Docker (optional but recommended)

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork locally**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Alpha-Newton.git
   cd Alpha-Newton
   ```

3. **Add the upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/Alpha-Newton.git
   ```

4. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

6. **Verify your setup**
   ```bash
   python test_pipeline.py
   ```

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

When filing an issue, include:

- **Clear title** describing the problem
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Environment details** (OS, Python version, GPU, CUDA version)
- **Error messages** and stack traces
- **Config files** used (sanitize any sensitive info)

### Suggesting Features

We love feature suggestions! When proposing a feature:

- **Explain the problem** you're trying to solve
- **Describe the solution** you'd like
- **Consider alternatives** you've thought about
- **Provide context** — how would this help other users?

### Code Contributions

#### Good First Issues

Look for issues labeled `good first issue` — these are great for newcomers!

#### Areas We Need Help

- **New dataset formats** — Add parsers for more conversation formats
- **Training algorithms** — Implement RLHF, GRPO, PPO, etc.
- **Optimizations** — Improve training speed and memory efficiency
- **Documentation** — Tutorials, examples, API docs
- **Testing** — Unit tests, integration tests
- **Multi-GPU support** — DeepSpeed, FSDP integration

---

## Development Workflow

### 1. Create a Branch

```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main

# Create your feature branch
git checkout -b feature/my-feature
# or for bugs: git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clean, readable code
- Add docstrings to new functions
- Update relevant documentation
- Add tests if applicable

### 3. Test Your Changes

```bash
# Run the test pipeline
python test_pipeline.py

# Test with a small training run
python src/entrypoints/train_sft.py --exp qwen3_600M_base_sft_test
```

### 4. Commit Your Changes

Write clear, concise commit messages:

```bash
# Good examples
git commit -m "Add support for Alpaca dataset format"
git commit -m "Fix memory leak in sequence packing"
git commit -m "Improve DPO trainer documentation"

# Bad examples
git commit -m "Fix stuff"
git commit -m "WIP"
git commit -m "Changes"
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/my-feature
```

Then open a PR on GitHub with:

- **Clear title** summarizing the change
- **Description** of what and why
- **Link to related issue** (if any)
- **Screenshots** (for UI changes)

---

## Style Guidelines

### Python Code Style

We follow PEP 8 with some additions:

```python
# Use type hints
def build_dataset(tokenizer: AutoTokenizer, max_seq_len: int = 2048) -> dict:
    """
    Build a tokenized dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        max_seq_len: Maximum sequence length for truncation
        
    Returns:
        Dictionary with 'train' and 'eval' datasets
    """
    pass

# Use descriptive variable names
dataset_config = {...}  # Good
dc = {...}              # Bad

# Keep functions focused
# Good: One function, one purpose
def load_dataset(...): ...
def format_dataset(...): ...
def tokenize_dataset(...): ...

# Bad: One giant function doing everything
def prepare_everything(...): ...
```

### Config Style

```yaml
# Use descriptive names
run:
  name: "qwen3_600M_sft_openhermes"  # Good
  name: "exp1"                        # Bad

# Group related settings
training:
  # Batch settings
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  
  # Learning rate settings
  learning_rate: 2.0e-5
  warmup_steps: 500

# Add comments for non-obvious settings
training:
  beta: 0.1  # DPO temperature (higher = more conservative)
```

### Documentation Style

- Use clear, simple language
- Include code examples
- Keep explanations concise
- Update docs when changing behavior

---

## Adding New Components

### Adding a New Dataset Type

1. Create your builder in `src/builders/data.py`:

```python
from src.core.registry import register

@register("data", "my_format")
def build_my_format_dataset(
    train_path: str,
    tokenizer,
    max_seq_len: int,
    **kwargs
):
    """
    Build dataset from my custom format.
    
    Expected format:
        {"question": "...", "answer": "..."}
    """
    # Load data
    dataset = load_dataset(train_path)
    
    # Format to messages
    def format_sample(sample):
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
    
    dataset = dataset.map(format_sample)
    
    # ... tokenization logic ...
    
    return {"train": train_dataset, "eval": eval_dataset}
```

2. Use it in your config:

```yaml
data:
  type: "my_format"
  train_path: "path/to/data"
```

### Adding a New Trainer

1. Create your trainer in `src/builders/trainer.py`:

```python
@register("trainer", "my_trainer")
def build_my_trainer(model, tokenizer, dataset, training_cfg, wandb_cfg):
    # Your trainer implementation
    pass
```

2. Create an entrypoint in `src/entrypoints/`:

```python
# src/entrypoints/train_my_method.py
from src.core.config import Config
from src.core.registry import build

def main(exp_name: str):
    cfg = Config.from_experiment(exp_name)
    
    tokenizer = build("tokenizer", **cfg.tokenizer)
    model = build("model", **cfg.model)
    dataset = build("data", tokenizer=tokenizer, **cfg.data)
    
    trainer = build(
        "trainer",
        type="my_trainer",
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_cfg=cfg.training,
        wandb_cfg=cfg.wandb,
    )
    
    trainer.train()
```

---

## Community

### Getting Help

- **GitHub Issues** — For bugs and feature requests
- **Discussions** — For questions and general chat

### Recognition

Contributors are recognized in:

- The project README
- Release notes for significant contributions
- GitHub's contributor graph

---

## Thank You!

Your contributions make Alpha-Newton better for everyone. Whether it's a bug report, feature suggestion, documentation improvement, or code contribution — every bit helps!

Happy training! 🚀

