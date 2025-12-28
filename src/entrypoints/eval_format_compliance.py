import argparse
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def extract_thinking_tags(text: str) -> tuple[bool, bool, str]:
    has_think_open = "<think>" in text
    has_think_close = "</think>" in text

    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, text, re.DOTALL)
    thinking_content = match.group(1).strip() if match else ""

    return has_think_open, has_think_close, thinking_content


def extract_final_answer(text: str) -> str | None:
    after_think = text.split("</think>")[-1] if "</think>" in text else text

    answer_patterns = [
        r"(?:answer is|answer:|=)\s*([+-]?\d+(?:\.\d+)?)",
        r"\\boxed\{([^}]+)\}",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, after_think, re.IGNORECASE)
        if match:
            return match.group(1)

    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", after_think)
    if numbers:
        return numbers[-1]

    return None


def evaluate_format_compliance(
    model_path: str,
    checkpoint: str = None,
    dataset_name: str = "openai/gsm8k",
    num_samples: int = 100,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    base_path = Path(model_path)

    if checkpoint:
        load_path = base_path / checkpoint
    else:
        load_path = base_path

    print(f"Loading model from: {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, "main", split="test")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    results = {
        "total": 0,
        "has_think_tags": 0,
        "has_complete_think_tags": 0,
        "has_answer": 0,
        "correct": 0,
        "avg_length": 0,
        "samples": []
    }

    total_length = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        question = sample["question"]
        ground_truth = sample["answer"].split("####")[-1].strip()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful math assistant. Show your reasoning using <think></think> tags, then provide your final answer."
            },
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        has_open, has_close, thinking_content = extract_thinking_tags(response)
        extracted_answer = extract_final_answer(response)

        results["total"] += 1
        total_length += len(tokenizer.encode(response))

        if has_open or has_close:
            results["has_think_tags"] += 1

        if has_open and has_close:
            results["has_complete_think_tags"] += 1

        if extracted_answer is not None:
            results["has_answer"] += 1

            if extracted_answer == ground_truth:
                results["correct"] += 1

        results["samples"].append({
            "question": question[:100],
            "response": response[:200],
            "has_think_tags": has_open and has_close,
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "correct": extracted_answer == ground_truth if extracted_answer else False,
        })

    results["avg_length"] = total_length / results["total"] if results["total"] > 0 else 0

    print("\n" + "="*60)
    print("FORMAT COMPLIANCE EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {load_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {results['total']}")
    print()
    print(f"Format Compliance Rate: {results['has_complete_think_tags'] / results['total'] * 100:.1f}%")
    print(f"  - Has any <think> tags: {results['has_think_tags'] / results['total'] * 100:.1f}%")
    print(f"  - Has complete <think></think>: {results['has_complete_think_tags'] / results['total'] * 100:.1f}%")
    print()
    print(f"Answer Extraction Rate: {results['has_answer'] / results['total'] * 100:.1f}%")
    print(f"Accuracy: {results['correct'] / results['total'] * 100:.1f}%")
    print()
    print(f"Average Output Length: {results['avg_length']:.0f} tokens")
    print()

    print("Sample Outputs:")
    for i, sample in enumerate(results["samples"][:5]):
        print(f"\n[{i+1}] {sample['question']}...")
        print(f"    Response: {sample['response']}...")
        print(f"    Tags: {sample['has_think_tags']} | Answer: {sample['extracted_answer']} | GT: {sample['ground_truth']} | Correct: {sample['correct']}")

    print("\n" + "="*60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate format compliance for thinking models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    evaluate_format_compliance(
        model_path=args.model,
        checkpoint=args.checkpoint,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
