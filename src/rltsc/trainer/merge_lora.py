import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

EXPORT = "models/sftqwen3/global_step_352_hf"


ADAPTER = os.environ.get("ADAPTER", os.path.join(EXPORT, "lora_adapter"))


OUT = "models/sftqwen2/Qwen2.5-3B-SFT"


def load_base_model(path: str):
    common_kwargs = dict(
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    try:
        return AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
            **common_kwargs,
        )

    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            **common_kwargs,
        )


model = load_base_model(EXPORT)


model = PeftModel.from_pretrained(model, ADAPTER)


model = model.merge_and_unload()


try:
    model.config.dtype = "bfloat16"


except Exception:
    pass


if hasattr(model.config, "torch_dtype"):
    model.config.torch_dtype = torch.bfloat16


os.makedirs(OUT, exist_ok=True)


model.save_pretrained(
    OUT,
    safe_serialization=True,
    max_shard_size="4GB",
)


tokenizer = AutoTokenizer.from_pretrained(EXPORT, trust_remote_code=True)


tokenizer.save_pretrained(OUT)


try:
    GenerationConfig.from_pretrained(EXPORT).save_pretrained(OUT)


except Exception:
    pass


for name in ("chat_template.jinja",):
    src = os.path.join(EXPORT, name)

    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, name))


print(f"Saved merged full model to: {OUT}")
