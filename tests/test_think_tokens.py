from transformers import AutoTokenizer


def test_think_tokens():
    print("Testing Qwen3-0.6B-Base tokenizer for <think> tags...\n")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    test_text = "<think>\nStep 1: Add 2 + 2\nStep 2: Result is 4\n</think>\n\nThe answer is 4"

    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"Original text:\n{test_text}\n")
    print(f"Token IDs: {tokens}\n")
    print(f"Number of tokens: {len(tokens)}\n")
    print(f"Decoded text:\n{decoded}\n")

    vocab = tokenizer.get_vocab()
    think_tokens = {k: v for k, v in vocab.items() if "think" in k.lower()}

    print("Think-related tokens in vocabulary:")
    for token, token_id in sorted(think_tokens.items(), key=lambda x: x[1]):
        print(f"  {token!r}: {token_id}")

    print("\nChecking specific tokens:")
    for tok in ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        if tok in vocab:
            print(f"  ✓ {tok!r} exists (ID: {vocab[tok]})")
        else:
            print(f"  ✗ {tok!r} not found")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Show your reasoning using <think></think> tags."
        },
        {
            "role": "user",
            "content": "What is 2+2?"
        },
        {
            "role": "assistant",
            "content": "<think>\nStep 1: Add 2 + 2\nStep 2: Result is 4\n</think>\n\nThe answer is 4"
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\n" + "="*60)
    print("Chat template output:")
    print("="*60)
    print(formatted)
    print("="*60)

    formatted_tokens = tokenizer.encode(formatted)
    print(f"\nFormatted chat tokens: {len(formatted_tokens)} tokens")


if __name__ == "__main__":
    test_think_tokens()
