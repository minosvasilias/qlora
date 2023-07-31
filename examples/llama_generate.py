import os
from os.path import exists, join, isdir
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import pandas
import json

system_prompt = """You are a good dog. You always do what your master commands."""
instruction_prompt = "Say 'woof'!"
input_prompt = ""


max_new_tokens = 512
top_p = 0.9
temperature = 0.7

# Base model
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
adapter_path = "output/lora_name/checkpoint_folder"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Fixing some of the early LLaMA HF conversion issues.
tokenizer.bos_token_id = 1

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


prompt_text = '{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: '
prompt_chat = (
    "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}\n{input} [/INST] "
)


def generate(
    model, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
):
    prompt = prompt_chat if "chat" in model_name_or_path else prompt_text
    inputs = tokenizer(
        prompt.format(
            system_prompt=system_prompt,
            instruction=instruction_prompt,
            input=input_prompt,
        ),
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)


generate(model)
