from transformers import AutoTokenizer
from src.core.registry import register

@register("tokenizer", "hf")
def build_tokenizer(id: str, padding_side: str, truncation_side: str, use_fast: bool):
    tok = AutoTokenizer.from_pretrained(id, use_fast=use_fast)
    tok.padding_side = padding_side
    tok.truncation_side = truncation_side
    return tok
