
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import torch


#warning :#you need more space for running this code because it will store model as same original size
#https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge?source=post_page-----3204119f0426--------------------------------

model_name = "Llama-2-7b-chat-hf"
adapter_model1 = "Llama-2-7b-chat-hf-finetune-omniscien-finetuned-dataset"

tokenizer = AutoTokenizer.from_pretrained(model_name)

#Load the base model with default precision
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

#Load and activate the adapter on top of the base model
model = PeftModel.from_pretrained(model, adapter_model1)

#Merge the adapter with the base model
model = model.merge_and_unload()

#Save the merged model in a directory "./naive_merge/" in the safetensors format
model.save_pretrained("./naive_merge_/", safe_serialization=True)
tokenizer.save_pretrained("./naive_merge/")

model_name =  "naive_merge"
adapter_model2 = "Llama-2-7b-chat-hf-finetune-finetuned-dataset-phrase-32-qlora-with-weight-decay-on-phrase-data-decreasd-lr"
#Load the base model with default precision
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

#Load and activate the adapter on top of the base model
model = PeftModel.from_pretrained(model, adapter_model2)

#Merge the adapter with the base model
model = model.merge_and_unload()

#Save the merged model in a directory "./naive_merge/" in the safetensors format
model.save_pretrained("./naive_merge_ver2/", safe_serialization=True)
tokenizer.save_pretrained("./naive_merge_ver2/")




# Call model you can reference below 
# compute_dtype = getattr(torch, "float16")
# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=True,
# )
# model = AutoModelForCausalLM.from_pretrained("./naive_merge/", quantization_config=bnb_config, device_map={"": 0})

