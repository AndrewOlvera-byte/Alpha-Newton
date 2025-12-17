from transformers import AutoTokenizer
from src.core.registry import register

@register("tokenizer", "hf")
def build_tokenizer(id: str, padding_side: str, truncation_side: str, use_fast: bool):
    tok = AutoTokenizer.from_pretrained(id, use_fast=use_fast)
    tok.padding_side = padding_side
    tok.truncation_side = truncation_side

    # Many base LMs don't define a pad token; for batching we fall back to EOS.
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    return tok
