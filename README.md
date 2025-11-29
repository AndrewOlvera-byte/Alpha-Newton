# Alpha-Newton

A modular LLM post-training framework for building instruction-tuned and reasoning-capable language models.

---

## 🚀 Quick Start

### Initial Setup

```bash
# Build and start the Docker container
docker compose build
docker compose up -d

# Verify GPU access
docker compose exec alpha-newton nvidia-smi
```

---

## 📋 Training Commands

### Phase 1: Instruct Model (General + Tools)

Train a helpful, safe model with tool calling capabilities.

```bash
# Full SFT training (70% OpenHermes + 30% Glaive, ~3-4 hours)
docker compose exec alpha-newton python src/entrypoints/train_sft.py --exp qwen3_600M_sft_mixed_optimized

# Then DPO for alignment (~30-45 min)
docker compose exec alpha-newton python src/entrypoints/train_dpo.py \
  --exp qwen3_600M_dpo_anthropic_hh
```

**Result:** `outputs/qwen3_600M_dpo_anthropic_hh/` - Ready-to-use instruct model

---

### Phase 2: Thinking Model (Reasoning)

Add chain-of-thought reasoning with `<think>` tags.

```bash
# SFT on reasoning traces (~2-3 hours)
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp qwen3_600M_sft_thinking
```

**Result:** `outputs/qwen3_600M_sft_thinking/` - Reasoning-capable model

---

### Quick Tests (Verify Setup)

```bash
# Test SFT pipeline (100 steps, ~5 min)
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp qwen3_600M_base_sft_test

# Test DPO pipeline (50 steps, ~3 min)
docker compose exec alpha-newton python src/entrypoints/train_dpo.py \
  --exp qwen3_600M_dpo_test

# Test all components (no training)
docker compose exec alpha-newton python test_pipeline.py
```

---

## 💬 Chat Interface

### Launch Web UI

```bash
# Chat with your trained model
docker compose exec alpha-newton python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_dpo_anthropic_hh

# Or use a specific checkpoint
docker compose exec alpha-newton python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_sft_mixed_optimized \
  --checkpoint checkpoint-5000

# Create public share link
docker compose exec alpha-newton python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_dpo_anthropic_hh \
  --share
```

**Access at:** http://localhost:7860

---

## 📊 Monitoring & Debugging

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 docker compose exec alpha-newton nvidia-smi

# Or one-time check
docker compose exec alpha-newton nvidia-smi
```

### Check Training Logs

```bash
# Follow training output
docker compose logs -f alpha-newton

# Check WandB dashboard
# → Login at https://wandb.ai
```

### Access Container Shell

```bash
# Interactive shell
docker compose exec alpha-newton bash

# Then you can run commands directly:
# python src/entrypoints/train_sft.py --exp ...
```

---

## 🛠️ Common Operations

### Restart Container

```bash
# Stop container
docker compose down

# Rebuild and start (after code/config changes)
docker compose build
docker compose up -d
```

### Clean Up

```bash
# Remove all containers and volumes
docker compose down -v

# Remove cached datasets (frees disk space)
rm -rf cache/

# Remove old checkpoints
rm -rf outputs/*/checkpoint-*
```

### Check Disk Space

```bash
docker compose exec alpha-newton df -h /workspace
```

---

## 📁 Directory Structure

```
Alpha-Newton/
├── configs/
│   ├── base/
│   │   └── common.yaml           # Shared hyperparameters
│   └── exp/
│       ├── qwen3_600M_sft_mixed_optimized.yaml
│       ├── qwen3_600M_dpo_anthropic_hh.yaml
│       └── qwen3_600M_sft_thinking.yaml
│
├── src/
│   ├── builders/                 # Model, data, trainer builders
│   ├── core/                     # Config system, registry
│   └── entrypoints/              # Training & chat scripts
│       ├── train_sft.py
│       ├── train_dpo.py
│       └── chat.py
│
├── outputs/                      # Trained models & checkpoints
├── cache/                        # Cached datasets
└── experiments/                  # WandB experiment data
```

---

## 🎯 Training Recipes

### Full Pipeline: Base → Instruct → Thinking

```bash
# 1. Instruct model with tools (~4 hours total)
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp qwen3_600M_sft_mixed_optimized

docker compose exec alpha-newton python src/entrypoints/train_dpo.py \
  --exp qwen3_600M_dpo_anthropic_hh

# 2. Add reasoning capabilities (~2-3 hours)
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp qwen3_600M_sft_thinking

# 3. Chat with final model
docker compose exec alpha-newton python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_sft_thinking
```

### Just Instruct (No Thinking)

```bash
# Fastest path to a usable model (~4 hours)
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp qwen3_600M_sft_mixed_optimized

docker compose exec alpha-newton python src/entrypoints/train_dpo.py \
  --exp qwen3_600M_dpo_anthropic_hh

# Use it
docker compose exec alpha-newton python src/entrypoints/chat.py \
  --model outputs/qwen3_600M_dpo_anthropic_hh
```

---

## 🔧 Configuration

### Create Custom Training Config

```bash
# Copy an existing config
cp configs/exp/qwen3_600M_sft_mixed_optimized.yaml \
   configs/exp/my_custom_run.yaml

# Edit the config
nano configs/exp/my_custom_run.yaml

# Run training
docker compose exec alpha-newton python src/entrypoints/train_sft.py \
  --exp my_custom_run
```

### Key Hyperparameters

**SFT (Supervised Fine-Tuning):**
- `learning_rate: 2.0e-5` - Standard for small models
- `max_steps: 10000` - ~100k examples, ~3-4 hours
- `per_device_train_batch_size: 8` - Optimized for 16GB VRAM

**DPO (Direct Preference Optimization):**
- `learning_rate: 5.0e-7` - Much lower than SFT (40x)
- `max_steps: 1000` - Converges quickly
- `beta: 0.1` - DPO temperature (0.1 = standard)

---

## 📚 Documentation

- **[DATASET_SCHEMA_GUIDE.md](DATASET_SCHEMA_GUIDE.md)** - How datasets are processed
- **[THINKING_MODEL_ROADMAP.md](THINKING_MODEL_ROADMAP.md)** - Path to AIMO competition
- **[LLM_POST_TRAINING_GUIDE.md](LLM_POST_TRAINING_GUIDE.md)** - Complete training guide
- **[PIPELINE_DEEP_DIVE.md](PIPELINE_DEEP_DIVE.md)** - Technical architecture
- **[DOCKER_USAGE.md](DOCKER_USAGE.md)** - Docker setup details
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - Experiment tracking

---

## ⚙️ Hardware Requirements

**Minimum:**
- GPU: 12GB VRAM (RTX 3060, RTX 4070)
- RAM: 16GB system memory
- Storage: 50GB free space

**Recommended (Current Setup):**
- GPU: 16GB VRAM (RTX 5070 Ti, RTX 4080)
- RAM: 32GB system memory
- Storage: 100GB free space

**For Larger Models:**
- GPU: 24GB VRAM (RTX 4090, A5000)
- Consider LoRA or quantization

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size in config:
per_device_train_batch_size: 4  # Instead of 8
gradient_accumulation_steps: 8  # Increase to maintain effective batch
```

### Port 7860 Not Accessible

```bash
# Restart container with port mapping
docker compose down
docker compose up -d

# Check if port is mapped
docker ps | grep alpha-newton
```

### Training Stuck at Data Loading

```bash
# Check if datasets are downloading
docker compose exec alpha-newton ls -lh cache/datasets/

# Increase num_proc if needed
num_proc: 12  # Use more CPU cores
```

### Model Quality Issues

```bash
# Check training loss in WandB
# → Should decrease steadily
# → Eval loss should track train loss

# Try lower learning rate if unstable
learning_rate: 1.0e-5  # Half of default

# Or train longer
max_steps: 15000  # More steps
```

---

## 🤝 Contributing

This is a research project. See the code for implementation details.

### Adding New Features

1. Add builder in `src/builders/`
2. Register with `@register()`
3. Create config in `configs/exp/`
4. Test with quick run

---

## 📖 Model Capabilities by Phase

| Capability | After SFT | After DPO | After Thinking |
|------------|-----------|-----------|----------------|
| **Chat** | ✅ Good | ✅ Better | ✅ Better |
| **Safety** | ⚠️ Basic | ✅ Strong | ✅ Strong |
| **Math (GSM8K)** | 30-40% | 35-45% | 60-80% |
| **Code** | ✅ Basic | ✅ Basic | ✅ Good |
| **Tool Use** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Reasoning** | ⚠️ Implicit | ⚠️ Implicit | ✅ Explicit |

---

## 🎯 Quick Reference: All Commands

```bash
# Setup
docker compose build && docker compose up -d

# SFT Training
docker compose exec alpha-newton python src/entrypoints/train_sft.py --exp qwen3_600M_sft_mixed_optimized

# DPO Training
docker compose exec alpha-newton python src/entrypoints/train_dpo.py --exp qwen3_600M_dpo_anthropic_hh

# Chat Interface
docker compose exec alpha-newton python src/entrypoints/chat.py --model outputs/qwen3_600M_dpo_anthropic_hh

# Monitor GPU
watch -n 1 docker compose exec alpha-newton nvidia-smi

# Shell Access
docker compose exec alpha-newton bash

# Stop
docker compose down
```

---

## 📊 Expected Timeline (RTX 5070 Ti)

| Task | Time | Output |
|------|------|--------|
| **SFT (Mixed)** | 3-4 hours | `outputs/qwen3_600M_sft_mixed_optimized/` |
| **DPO (Alignment)** | 30-45 min | `outputs/qwen3_600M_dpo_anthropic_hh/` |
| **SFT (Thinking)** | 2-3 hours | `outputs/qwen3_600M_sft_thinking/` |
| **Total Pipeline** | ~6-7 hours | Full reasoning model |

---

## 🔗 Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen Models](https://huggingface.co/Qwen)
- [OpenHermes Dataset](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [Glaive Function Calling](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

---

## 📝 License

Research project - see code for details.

---

**Built for training helpful, safe, and reasoning-capable language models.** 🚀
