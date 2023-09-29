from json import load
import os
import streamlit as st
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory
import time

import openpyxl
from openpyxl.styles import Font
from openpyxl.worksheet.hyperlink import Hyperlink
from datetime import datetime

import psutil

import torch
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, T5Tokenizer, AutoModel, T5ForConditionalGeneration, GPT2TokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from langchain.callbacks.base import BaseCallbackHandler

from peft import PeftModel
from peft.tuners.lora import LoraLayer

#upload document section
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import DataFrameLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pathlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

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


# langchain section
model_quantization_dict = {
    'fastchat-t5-3b-v1.0': ['bitsandbytes', ],
    'Llama-2-7b-chat-hf': ['bitsandbytes', 'autoGPTQ', 'llamaCpp'],
    'open_llama_7b': ['bitsandbytes', 'autoGPTQ', 'llamaCpp'],
}

model_quantization_finetune_dict = {'Llama-2-7b-chat-hf-finetune-omniscien-finetuned-dataset' : ["finetune"]
}

model_metadata = {
    "fastchat-t5-3b-v1.0": {"max_token": 512, 
                        "task" : "text2text-generation",
                        "temperature" : 0},
    "Llama-2-7b-chat-hf": {"max_token" : 2048, 
                        "task" : "text-generation",
                        "temperature" : 0.1},
    "open_llama_7b"     : {"max_token" : 32, 
                        "task" : "text-generation",
                        "temperature" : 0},
    "Llama-2-7b-chat-hf-finetune-omniscien-finetuned-dataset" : {"max_token" : 2048,
                                                                "task" : "text-generation",
                                                                "temperature" : 0}
}

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer. Please don't show unhelpful answers. If the question contains wrong spelling of person name, correct wrong spelling and show it in answer. 

Context : {context}
Chat history: {chat_history}
Question : {question}

Please check the question is relate with chat history first before answer if not please ignore chat history and use the current context and current question to generate the answer only
The answer should consist of at least 1 sentence for short questions or 7 sentences for more detailed qeustions. Only returns the helpful and reasonable answer below and nothing else.
No need to return the question. I just want answer. Please don't show unhelpful answers.
Helpful answer:
"""

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Please check the question is relate with chat history if not should standalone question should be the same with follow up input
Standalone question:"""
#CONDENSE_QUESTION_PROMPT_1 = PromptTemplate.from_template(_template)

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context',
                                                                              'question',
                                                                              'chat_history'])
    return prompt

def check_model_quantization(model_name, quantization_option):

    if quantization_option != 'No quantization' or quantization_option != "finetune":
        quantize_option = model_quantization_dict[model_name]
        if quantization_option in quantize_option:
            st.markdown(f"**{model_name}** is compatible with **{quantization_option}** ✅✅")
        else:
            st.markdown(f"**{model_name}** is *NOT* compatible with **{quantization_option}** ❌ ❌  ")
            st.markdown(f"The compatible option for this model is {quantize_option}")
    
    elif quantization_option == "finetune":
        st.markdown((f"**{model_name}** was finetuned ✅✅"))
    else:
        st.markdown((f"**{model_name}** without quantization ✅✅"))

@st.cache_resource
def load_model_no_quantize(model_option, model_metadata):
    if model_option == "fastchat-t5-3b-v1.0" :
        model = T5ForConditionalGeneration.from_pretrained(model_option,
                                        torch_dtype=torch.float16,
                                        device_map= 'auto')
    elif model_option == "Llama-2-7b-chat-hf" or model_option == "open_llama_7b":
        model = LlamaForCausalLM.from_pretrained(model_option,
                                                device_map= 'auto',
                                                torch_dtype=torch.float16)
    tokenizer =  AutoTokenizer.from_pretrained(model_option)
    pipe = pipeline(
            model_metadata[model_option]["task"], 
            model = model, 
            tokenizer = tokenizer,
            max_new_tokens = model_metadata[model_option]["max_token"], 
            max_length = model_metadata[model_option]["max_token"],
            model_kwargs ={"temperature":model_metadata[model_option]["temperature"]}
            )
    llm_hf = HuggingFacePipeline(pipeline=pipe)
    return llm_hf


@st.cache_resource
def load_llama2_bitsandbytes(model_option, model_metadata):
    if model_option == "fastchat-t5-3b-v1.0" :
        bitsandbyte_config = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.float16)

        model = T5ForConditionalGeneration.from_pretrained(model_option,
                                                quantization_config = bitsandbyte_config,
                                                device_map= 'auto')
    elif model_option == "Llama-2-7b-chat-hf" or model_option == "open_llama_7b":
        bitsandbyte_config = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.float16)

        model = LlamaForCausalLM.from_pretrained(model_option,
                                                quantization_config = bitsandbyte_config,
                                                device_map= 'auto')
    tokenizer =  AutoTokenizer.from_pretrained(model_option)
    pipe = pipeline(
            model_metadata[model_option]["task"], 
            model = model, 
            tokenizer = tokenizer,
            max_new_tokens= model_metadata[model_option]["max_token"], 
            max_length = model_metadata[model_option]["max_token"],
            model_kwargs ={"temperature":model_metadata[model_option]["temperature"]}
            )
    llm_hf = HuggingFacePipeline(pipeline=pipe)

    return llm_hf


@st.cache_resource
def load_llama2_GPTQ(model_option, model_metadata):
    core_model_name = model_option + "-GPTQ-4bit"    

    model = AutoModelForCausalLM.from_pretrained(core_model_name,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")

    tokenizer = AutoTokenizer.from_pretrained(core_model_name, use_fast=True)
    
    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = model_metadata[model_option]["max_token"],
        temperature = model_metadata[model_option]["temperature"]
    )

    llm_hf_GPTQ = HuggingFacePipeline(pipeline=pipe)
    return llm_hf_GPTQ


@st.cache_resource
def load_llama2_llamaCpp(model_option, model_metadata):
    core_model_name = model_option + ".gguf.q4_0.bin"
    n_gpu_layers = 32
    n_batch = 512
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=core_model_name,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,n_ctx = 4096, temperature = 0.1, max_tokens = model_metadata[model_option]["max_token"]
    )
    return llm

@st.cache_resource
def load_llama2_finetune(model_option, model_metadata):
    model_name_or_path = "Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        #torch_dtype=torch.float16,
        device_map="auto",
        #load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            #bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )
    model = PeftModel.from_pretrained(model, model_option)
    pipe = pipeline(
            "text-generation", 
            model = model, 
            tokenizer = tokenizer,
            max_new_tokens = 2048, 
            #max_length = 256,
            model_kwargs ={"temperature": 0, "repetition_penalty" : 1.3}
            )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def convert_to_website_format(urls):
    # Remove any '.html' at the end of the URL
    urls = re.sub(r'\.html$', '', urls)
    # Check if the URL starts with 'www.' or 'http://'
    if not re.match(r'(www\.|http://)', urls):
        urls = 'https://' + urls
    if '/index' in urls:
        urls = urls.split('/index')[0]
    match = re.match(r'^([^ ]+)', urls)
    if match:
        urls = match.group(1)
    return urls

def regex_source(answer):
    convert_urls = convert_to_website_format(answer)
    return convert_urls


@st.cache_resource 
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "thenlper/gte-base",
                                       model_kwargs = {'device': 'cuda'})
    return embeddings

def log_to_excel(file_path, demo_version, model, quantization, question, context, answer, url_lst, response_time):
    try:
        # Load existing workbook or create a new one
        try:
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
        except FileNotFoundError:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(["Timestamp", "Demo Version","Model", "Quantization", "Question", "Context", "Answer", "URL", "Response Time"])

        # Add new entry to the sheet
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        urls_formatted = '\n'.join(url_lst)

        if question is not None and answer is not None and urls_formatted is not None and response_time is not None:
            sheet.append([timestamp, demo_version, model, quantization, question, context, answer, urls_formatted, response_time])
            workbook.save(file_path)
            print("Logged successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / 1024 / 1024 /1024  # Convert bytes to megabytes
    print(f"Memory Usage: {memory_usage:.2f} GB")


def measure_storage_space():
    script_path = os.path.abspath(__file__)
    disk_usage = psutil.disk_usage(os.path.dirname(script_path))
    used_space = disk_usage.used / 1024 / 1024 / 1024  # Convert bytes to gigabytes
    print(f"Used Disk Space by Current Script: {used_space:.2f} GB")

def call_llm(quantize_option, model_option):
    if quantize_option == "No quantization" :
        llm = load_model_no_quantize(model_option, model_metadata)
        return llm

    elif quantize_option == 'llamaCpp':
        llm = load_llama2_llamaCpp(model_option, model_metadata)
        return llm

    elif quantize_option == 'autoGPTQ':
        llm = load_llama2_GPTQ(model_option, model_metadata)
        return llm 
    
    elif quantize_option == 'bitsandbytes':
        llm = load_llama2_bitsandbytes(model_option, model_metadata)
        return llm

    elif quantize_option == "finetune":
        llm = load_llama2_finetune(model_option, model_metadata)
        return llm

def clean_chat_history():
    st.session_state.messages = []
    msgs.clear()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


def main():
    # Initialize chat history
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    print(msgs)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Display chat messages from history on app rerun


    fdate = datetime.now().strftime('%Y-%m-%d')
    file_path = "docqa_v-2.0_logs_"+fdate+".xlsx"
    demo_version = "2.0"
    
    global index
    st.subheader("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES")

    # model_option = "fastchat-t5-3b-v1.0"
    # quantize_option = "No quantization"
    model_type = st.selectbox("Document option",("Base-model", "Finetuned"))
    if model_type == "Base-model" :
        model_option = "fastchat-t5-3b-v1.0"
        quantize_option = "No quantization"
        with st.form(key='basemodel'):
            col1, col2, col3 = st.columns([3,1,1])
            with col1:
                model_option = st.selectbox(
                    'Base model',
                    options =  list(model_quantization_dict.keys()))
            with col2:
                quantize_option = st.selectbox(
                    "4-bit Quantization",
                    options = ['No quantization', 'bitsandbytes', 'llamaCpp', 'autoGPTQ']
                )
            with col3:  
                memory_option = st.selectbox("Add memory",
                                options = ["No memory", "memory"]
                )
            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                st.write("model_option: ", model_option, "quantize_option: ", quantize_option) 

    elif model_type == "Finetuned":
        model_option = "Llama-2-7b-chat-hf-finetune-omniscien-finetuned-dataset"
        quantize_option = "finetune"
        with st.form(key = 'finetuned'):
            col1, col2 = st.columns([3,1])
            with col1:
                model_option = st.selectbox(
                        'Finetune model',
                        options =  list(model_quantization_finetune_dict.keys())) 
            with col2:
                memory_option = st.selectbox("Add memory",
                                options = ["No memory", "memory"]
                )
            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                st.write("model_option: ", model_option, "quantize_option: ", quantize_option) 

    # chat memory status refer with ConversationBufferWindowMemory
    memory_status = 0
    if memory_option == "memory":
      memory_status = 1
    elif memory_option == "No memory":
      memory_status = 0

    #check_model_quantization(model_option, quantize_option)
    st.markdown("""---""") 

    #create vector 

    # Define innitial use omniscien vector 
    data = []
    # DB_FAISS_UPLOAD_PATH = "vectorstores/db_faiss"
    DB_FAISS_OMNISCIEN_PATH = "./vectorstores_clean_doc_gte-base/db_faiss"
    embeddings = load_embeddings()
    db = FAISS.load_local(DB_FAISS_OMNISCIEN_PATH, embeddings)

    #call model
    llm = call_llm(quantize_option, model_option)
    qa_prompt = set_custom_prompt(custom_prompt_template)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt = qa_prompt)
    memory = ConversationBufferWindowMemory(k = memory_status,  chat_memory=msgs, return_messages=True,  input_key= 'question', output_key='answer', memory_key="chat_history")
    
    qa_chain = ConversationalRetrievalChain(
        retriever =db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k':5, "score_threshold": 0.7 }), 
        question_generator=question_generator,
        #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory = memory,
        get_chat_history=lambda h :h)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        # Accept user input
    if query := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        start = time.time()
        context = ""

        if quantize_option == 'llamaCpp':
            with st.chat_message("assistant"):
                answer = st.empty()
                stream_handler = StreamHandler(answer, initial_text="")
                response = qa_chain({"question": query},callbacks=[stream_handler])

        else:
            response = qa_chain({'question': query})

        url_list = set([i.metadata['source']  for i in response['source_documents']])
        print(f"condensed quesion : {question_generator.run({'chat_history': response['chat_history'], 'question' : query})}")
        
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")
        measure_memory_usage()

        log_to_excel(file_path, demo_version, model_option, quantize_option, query, context, response["answer"], url_list, str(int(end-start))+" sec")

        if quantize_option != 'llamaCpp':
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})


        with st.expander("See the related documents"):
            for count, url in enumerate(url_list):
                url_reg = regex_source(url)
                st.write(str(count+1)+":", url_reg)
    view_messages = st.expander("View the message contents in session state")
    with view_messages:
        view_messages.json(st.session_state.langchain_messages)

    clear_button = st.button("Start new convo")
    if clear_button :
        st.session_state.messages = []
        qa_chain.memory.chat_memory.clear() 
        

if __name__ == '__main__':
        main()
