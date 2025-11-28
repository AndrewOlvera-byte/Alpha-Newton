from src.core.registry import register
from transformers import AutoModelForCausalLM

@register("model", "hf_causal")
def build_hf_model(id: str, revision: str, dtype: str, trust_remote_code: bool, load_in_4bit: bool, gradient_checkpointing: bool):

    model = AutoModelForCausalLM.from_pretrained(
        id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model
