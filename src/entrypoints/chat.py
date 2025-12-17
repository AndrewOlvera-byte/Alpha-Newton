#!/usr/bin/env python3
"""
Interactive chat interface for trained models using Gradio

Usage:
    python src/entrypoints/chat.py --model outputs/qwen3_600M_base_sft_test
    python src/entrypoints/chat.py --model outputs/qwen3_600M_base_sft_test --checkpoint checkpoint-100
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


def load_model(model_path: str, checkpoint: str = None):
    """Load trained model and tokenizer"""
    base_path = Path(model_path)

    # If checkpoint specified, load from checkpoint subdirectory
    if checkpoint:
        load_path = base_path / checkpoint
    else:
        load_path = base_path

    if not load_path.exists():
        raise ValueError(f"Model path does not exist: {load_path}")

    # Check if tokenizer files exist
    required_files = ["tokenizer.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not (load_path / f).exists()]

    if missing_files:
        # List available checkpoints
        checkpoints = sorted([d.name for d in base_path.glob("checkpoint-*") if d.is_dir()])
        error_msg = f"Tokenizer files not found in: {load_path}\n"

        if checkpoints:
            error_msg += f"\nAvailable checkpoints:\n"
            for cp in checkpoints:
                error_msg += f"  - {cp}\n"
            error_msg += f"\nTry: --checkpoint {checkpoints[-1]}"
        else:
            error_msg += "\nNo checkpoints found. Run training first."

        raise ValueError(error_msg)

    print(f"Loading model from: {load_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path, fix_mistral_regex=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f" Model loaded: {model.config.model_type}")
    print(f" Device: {model.device}")

    return model, tokenizer


def generate_response(
    message: str,
    history: list,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate response from model

    Args:
        message: User's current message
        history: List of [user_msg, bot_msg] pairs
        model: Loaded model
        tokenizer: Loaded tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    """
    # Build conversation history
    conversation = []
    for user_msg, bot_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": bot_msg})

    # Add current message
    conversation.append({"role": "user", "content": message})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        # Qwen-style chat templates use an explicit end-of-message token.
        # Include it as an EOS to prevent the model from "continuing the transcript"
        # into the next user/assistant turn.
        eos_token_ids = []
        if tokenizer.eos_token_id is not None:
            eos_token_ids.append(tokenizer.eos_token_id)

        for tok in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None) and tid not in eos_token_ids:
                eos_token_ids.append(tid)

        eos_arg = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=pad_id,
            eos_token_id=eos_arg,
        )

    # Decode only the new tokens (exclude the prompt)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return response


def create_chat_interface(model, tokenizer, model_name: str):
    """Create Gradio chat interface"""

    def respond(message, history, max_tokens, temperature, top_p):
        response = generate_response(
            message=message,
            history=history,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response

    # Create Gradio interface
    demo = gr.ChatInterface(
        fn=respond,
        title=f"Chat with {model_name}",
        description="Interactive chat interface for your fine-tuned model",
        additional_inputs=[
            gr.Slider(
                minimum=64,
                maximum=2048,
                value=512,
                step=64,
                label="Max New Tokens",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top P",
            ),
        ],
        examples=[
            ["Hello! How are you?", 512, 0.7, 0.9],
            ["Explain quantum computing in simple terms.", 512, 0.7, 0.9],
            ["Write a Python function to calculate fibonacci numbers.", 512, 0.7, 0.9],
        ],
        cache_examples=False,
    )

    return demo


def main(model_path: str, checkpoint: str = None, share: bool = False):
    """Launch chat interface"""

    print("=" * 60)
    print("Alpha-Newton Chat Interface")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(model_path, checkpoint)

    # Get model name for display
    model_name = Path(model_path).name
    if checkpoint:
        model_name = f"{model_name}/{checkpoint}"

    # Create and launch interface
    print(f"\nLaunching Gradio interface...")
    print(f"Model: {model_name}")

    demo = create_chat_interface(model, tokenizer, model_name)

    demo.launch(
        server_name="0.0.0.0",  # Allow external connections (for Docker)
        server_port=7860,
        share=share,  # Create public link if requested
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory (e.g., outputs/qwen3_600M_base_sft_test)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to load (e.g., checkpoint-100)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link (for remote access)",
    )

    args = parser.parse_args()
    main(model_path=args.model, checkpoint=args.checkpoint, share=args.share)
