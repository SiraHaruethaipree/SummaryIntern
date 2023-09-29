from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import torch

model_name = "Llama-2-7b-chat-hf"
adapter_model1 = "Llama-2-7b-chat-hf-finetune-omniscien-finetuned-dataset"
adapter_model2 = "Llama-2-7b-chat-hf-finetune-finetuned-dataset-phrase-32-qlora-with-weight-decay-on-phrase-data-decreasd-lr"

#4bit 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
)
model = PeftModel.from_pretrained(model, adapter_model1)

#Merge the adapter with the base model
model = model.merge_and_unload()

#Save the merged model in a directory "./naive_merge/" in the safetensors format
model.save_pretrained("./naive_merge_4-bit/", safe_serialization=True)
