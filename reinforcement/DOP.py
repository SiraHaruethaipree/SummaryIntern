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

print(torch.cuda.device_count())
print(torch.cuda.empty_cache())

model_path = "output_finetune_nf4"
dataset_path_rl = "Dahoas/full-hh-rlhf"
# dataset_rl = load_dataset(dataset_path_rl,split="train")
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

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
dataset_rl = dataset_finetune(dataset_path_rl, tokenizer, size=100)

def return_prompt_and_responses(samples) :
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
            for question in samples["prompt"]
        ],
        "chosen": samples["chosen"],   # rated better than k
        "rejected": samples["rejected"], # rated worse than j
    }

original_columns = dataset_rl.column_names

dataset_rl.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)

training_args=TrainingArguments(
    gradient_accumulation_steps=16,
    warmup_steps=2,#Number of steps used for a linear warmup from 0 to learning_rate.
    #max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_strategy="epoch",
    output_dir="./results",
    optim="adamw_hf",
    num_train_epochs=3,
    seed = 42,
    save_strategy="epoch",
    per_device_train_batch_size = 4,)

peft_config = LoraConfig(
    r= 8,
    lora_alpha= 32,
    lora_dropout= 0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoPeftModelForCausalLM.from_pretrained(
    model_path, # location of saved SFT model
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    is_trainable=True,
)
model_ref = AutoPeftModelForCausalLM.from_pretrained(
    model_path,  # same model as the main one
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
dpo_trainer = DPOTrainer(
    model = model,
    args=training_args,
    beta=0.1,
    train_dataset=dataset_rl,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
dpo_trainer.train()
dpo_trainer.save_model("output_reinforcement_nf4")