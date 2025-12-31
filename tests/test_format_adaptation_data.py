import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from src.builders.data import build_format_adaptation_dataset


def test_format_adaptation_data():
    print("Testing format adaptation data builder...\n")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    datasets_config = [
        {"path": "AI-MO/NuminaMath-CoT", "weight": 0.70},
        {"path": "microsoft/orca-math-word-problems-200k", "weight": 0.20},
        {"path": "HuggingFaceH4/no_robots", "weight": 0.10},
    ]

    print("Loading small subset (this may take a few minutes on first run)...\n")

    try:
        dataset = build_format_adaptation_dataset(
            datasets_config=datasets_config,
            max_seq_len=2048,
            num_proc=4,
            packing=False,
            tokenizer=tokenizer,
            max_token_filter=1800,
        )

        print(f"\nDataset loaded successfully!")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Eval samples: {len(dataset['eval'])}")

        print("\nInspecting first training sample:")
        sample = dataset['train'][0]
        print(f"  input_ids length: {len(sample['input_ids'])}")
        print(f"  labels length: {len(sample['labels'])}")

        decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print(f"\nDecoded sample (first 500 chars):")
        print(decoded[:500])

        if "<think>" in decoded:
            print("\n✓ <think> tag found in sample!")
        else:
            print("\n✗ Warning: <think> tag not found in first sample")

        if "</think>" in decoded:
            print("✓ </think> tag found in sample!")
        else:
            print("✗ Warning: </think> tag not found in first sample")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_format_adaptation_data()
