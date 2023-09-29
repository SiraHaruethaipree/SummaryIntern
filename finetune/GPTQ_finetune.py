#!pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/
#!pip install -q -U transformers peft accelerate optimum
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
import numpy as np
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers

import os
os.environ["WANDB_API_KEY"] = "49cce26d7f8bdfa8a093ba11fc5c3534f447ba83"

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

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj","o_proj","q_proj","v_proj"], #Wq , Wk, Wv , and Wo to refer to the query/key/value/outpu
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

# We disable the exllama kernel because training with exllama kernel is unstable. 
# To do that, we pass a `GPTQConfig` object with `disable_exllama=True`. This will overwrite the value stored in the config of the model.
def GPTQ_model(model_path, config):
    quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",revision="main", quantization_config=quantization_config_loading)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(model)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, tokenizer

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


if __name__ == '__main__':
    model_path = input("model path :")
    dataset_path = input("dataset path :")
    model, tokenizer = GPTQ_model(model_path, config)
    data = dataset_finetune(dataset_path, tokenizer, size=100)
    # needed for llama 2 tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        dataset_text_field= "text",
        args=training_args,)
    transformers.logging.set_verbosity_info()
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    trainer.train()
    trainer.save_model(f"{model_path}-finetune-{dataset_path}")