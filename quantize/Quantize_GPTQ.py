from optimum.gptq import GPTQQuantizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from torch.nn.parallel import DataParallel
import json
import torch.nn as nn
 

dataset_id = "wikitext2"
model_id = "Llama-2-7b-chat-hf"
save_folder = "Llama-2-7b-chat-hf-GPTQ-4bit"

def check_cuda():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(torch.cuda.get_device_name(i))
    return 

def quantize_process(model, tokenizer):  
    # GPTQ quantizer
    quantizer = GPTQQuantizer(bits=4, dataset=dataset_id, model_seqlen=2048) #reduce seqlen
    quantizer.quant_method = "gptq"
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False).save_pretrained(save_folder) # bug with fast tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map = "auto",) # we load the model in fp16 on purpose
    quantized_model.save_pretrained(save_folder, safe_serialization=True)

    # save quantize_config.json for TGI
    with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
    quantizer.disable_exllama = False
    json.dump(quantizer.to_dict(), f, indent=2)

    with open(os.path.join(save_folder, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
    config["quantization_config"]["disable_exllama"] = False
    with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("finish")

