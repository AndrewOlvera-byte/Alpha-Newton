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

