import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
import numpy as np
from trl import DPOTrainer, SFTTrainer
os.environ["WANDB_API_KEY"] = "49cce26d7f8bdfa8a093ba11fc5c3534f447ba83"

model_path = "Llama-2-7b-chat-hf"
training_args=TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,#Number of steps used for a linear warmup from 0 to learning_rate.
    #max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_strategy="epoch",
    output_dir="./results",
    optim="adamw_hf",
    num_train_epochs=3,
    seed = 42,
    save_strategy="epoch")
# load the base model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,        # "meta-llama/Llama-2-7b-hf"
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
base_model.config.use_cache = False

dataset_path = "timdettmers/openassistant-guanaco"
def dataset_finetune(dataset_path, tokenizer, size = None):
    #dataset_name = "timdettmers/openassistant-guanaco"
    dataset = load_dataset(dataset_path, split="train")
    np.random.seed(42)
    if size is not None :
        random_idx= np.random.randint(low= 0, high =len(dataset), size = size)
        slice_dataset = dataset.select(random_idx)
        print(len(slice_dataset))
        data = slice_dataset.map(lambda samples:tokenizer(samples["text"]), batched=True)
    else :
        data = dataset.map(lambda samples:tokenizer(samples["text"]), batched=True)
    return data
data = dataset_finetune(dataset_path, tokenizer, size=100)

tokenizer.pad_token = tokenizer.eos_token
# add LoRA layers on top of the quantized base model
peft_config = LoraConfig(
    r= 8,
    lora_alpha= 32,
    lora_dropout= 0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
trainer = SFTTrainer(
    model=base_model,
    train_dataset=data,
    peft_config=peft_config,
    #packing=True,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,         # HF Trainer arguments
    dataset_text_field= "text"
)
trainer.train()
trainer.save_model(f"output_finetune_nf4")