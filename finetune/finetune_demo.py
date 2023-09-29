
from json import load
import os
import streamlit as st
import re
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
from trl import SFTTrainer
import transformers

import time
import openpyxl
from openpyxl.styles import Font
from openpyxl.worksheet.hyperlink import Hyperlink
from datetime import datetime

import psutil

import torch
#from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

#from llm_experiment2 import *


# sidebar contents
with st.sidebar:
        st.title('DOC-QA DEMO ')
        st.markdown('''
        ## About        
        This app is an LLM-powered Doc-QA Demo built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [HuggingFace](https://huggingface.co/declare-lab/flan-alpaca-large)
        ''')
        st.subheader ('Made this app for testing Document Question Answering with Custom Crawled Webstie Data')

# model_quantization_dict = {
#     'fastchat-t5-3b-v1.0': ['bitsandbytes', ],
#     'Llama-2-7b-chat-hf': ['bitsandbytes', 'autoGPTQ', 'llamaCpp'],
#     'open_llama_7b': ['bitsandbytes', 'autoGPTQ', 'llamaCpp']
# }

model_metadata = {
    "fastchat-t5-3b-v1.0": {"max_token": 512, 
                        "task" : "text2text-generation",
                        "temperature" : 0},
    "Llama-2-7b-chat-hf": {"max_token" : 2048, 
                        "task" : "text-generation",
                        "temperature" : 0.1},
    "open_llama_7b"     : {"max_token" : 32, 
                        "task" : "text-generation",
                        "temperature" : 0}
}


training_args=TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,#Number of steps used for a linear warmup from 0 to learning_rate.
    #max_steps=10,
    learning_rate=2e-4,
    #fp16=True,
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

def GPTQ_model(model_option, config):
    model_path = model_option + "-GPTQ-4bit"
    quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",revision="main", quantization_config=quantization_config_loading)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(model)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, tokenizer

def NF4_model(model_option, config):
    model_path = model_option
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,        # "meta-llama/Llama-2-7b-hf"
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16)
    return model, tokenizer

def GUFF_model(model_option, config):
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

def check_method(model_option, quantize_option, config):
    if quantize_option == "bitsandbytes":
        model, tokenizer = NF4_model()
        return  model, tokenizer
    elif quantize_option == "llamaCpp" :
        model, tokenizer = GUFF_model()
        return  model, tokenizer
    elif quantize_option == "autoGPTQ" :
        model, tokenizer  = GPTQ_model(model_option, config)
        return  model, tokenizer

def main():
    
    global index
    st.subheader("Fine Tune LLM Model 4-bit")

    # model_option = "fastchat-t5-3b-v1.0"
    # quantize_option = "No quantization"
    with st.form(key='my_form'):
        col1, col2 = st.columns([3,1])
        with col1:
            model_option = st.selectbox(
                'Base model',
                options =  list(model_metadata.keys()))
        with col2:
            quantize_option = st.selectbox(
                "4-bit Quantization option",
                options = ['bitsandbytes', 'llamaCpp', 'autoGPTQ']
            )
        dataset = st.selectbox(
                "Finetune Dataset from huggingface",
                options = ["timdettmers/openassistant-guanaco"]
            )
        output_name = st.text_input(
        "Save file/folder name")

        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            # torch.cuda.empty_cache()
            st.write("model_option: ", model_option, "quantize_option: ", quantize_option)
            st.write("Fine-tune with", dataset, "File name ", output_name)
                # model_path = input("model path :")
                # dataset_path = input("dataset path :")
                # model, tokenizer = GPTQ_model(model_path, config)
                # data = dataset_finetune(dataset_path, tokenizer, size=100)
                # # needed for llama 2 tokenizer
                # tokenizer.pad_token = tokenizer.eos_token
                # trainer = SFTTrainer(
                #     model=model,
                #     train_dataset=data,
                #     peft_config=config,
                #     max_seq_length=2048,
                #     tokenizer=tokenizer,
                #     dataset_text_field= "text",
                #     args=training_args,)
                # transformers.logging.set_verbosity_info()
                # model.config.use_cache = False # silence the warnings. Please re-enable for inference!
                # trainer.train()
                # trainer.save_model(f"{model_path}-finetune-{dataset_path}")

if __name__ == '__main__':
        main()

#finetune
#NF4
#open_llama_7b
#fastchat-t5-3b-v1.0
#Llama-2-7b-chat-hf

#GPTQ
#fastchat-t5-3b-v1.0-GPTQ-4bit
#Llama-2-7b-chat-hf-GPTQ-4bit
#open_llama_7b-GPTQ-4bit

#GUFF
#open_llama_7b
#fastchat-t5-3b-v1.0
#Llama-2-7b-chat-hf