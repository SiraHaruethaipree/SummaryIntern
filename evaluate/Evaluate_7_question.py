#This python script is create for evaluate finetune model that already fintune
#with omniscien dataset and combine with RAG technique for evaluate.

import os
from os.path import exists, join, isdir
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

from json import load
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
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
from transformers import pipeline, T5Tokenizer, AutoModel, T5ForConditionalGeneration, GPT2TokenizerFast
from transformers import LlamaForCausalLM, LlamaTokenizer

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
#from llm_experiment2 import *
from langchain.callbacks.base import BaseCallbackHandler
import pandas as pd
import datetime

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

# Fixing some of the early LLaMA HF conversion issues.
#tokenizer.bos_token_id = 1

def load_basemodel(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
        )
    )
    pipe = pipeline(
            "text-generation", 
            model = model, 
            tokenizer = tokenizer,
            max_new_tokens = 256, 
            #max_length = 256,
            model_kwargs ={"temperature": 0, "repetition_penalty" : 1.3}
            )
    llm_hf = HuggingFacePipeline(pipeline=pipe)
    return llm_hf


def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "thenlper/gte-base",
                                       model_kwargs = {'device': 'cuda'})
    return embeddings


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

def main(): 
    #msgs = StreamlitChatMessageHistory(key="langchain_messages")
    #print(msgs)
    DB_FAISS_PATH = "./vectorstores_clean_doc_gte-base/db_faiss"
    embeddings = load_embeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    model_name_or_path = 'naive_merge_ver2'
    #adapter_path = 'Llama-2-7b-chat-hf-finetune-finetuned-dataset-phrase-32-qlora-with-weight-decay-on-phrase-data-decreasd-lr'
    #llm = call_llm(quantize_option, model_option)
    llm = load_basemodel(model_name_or_path)
    qa_prompt = set_custom_prompt(custom_prompt_template)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt = qa_prompt)
    memory = ConversationBufferWindowMemory(k = 0, return_messages=True,  input_key= 'question', output_key='answer', memory_key="chat_history")
    
    qa_chain = ConversationalRetrievalChain(
        retriever =db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k':5, "score_threshold": 0.7 }), 
        question_generator=question_generator,
        #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory = memory,
        get_chat_history=lambda h :h)

    #evaluate 
    #print(adapter_path)
    response_list = []
    question_list = []
    time_list = []
    url_list = []
    # ds = load_from_disk("finetuned-dataset-omniscien")
    # question_set = ds['test_ds']["question"]


    test_df = pd.read_csv("omniscien-test-7q.csv")
    question_set = test_df['Question'].tolist()


    print(f"your question lenght: {len(question_set)}")
    for i, question in enumerate(question_set):
        start = time.time()
        question_list.append(question)
        print(f"{i+1}. {question}")
        response = qa_chain({'question': question})
        print(response['answer'])
        response_list.append(response['answer'])
        end = time.time()
        timer = int(end-start)
        time_list.append(timer)
        print(f'Respone time: {timer} sec')
        urls_answer = set([i.metadata['source']  for i in response['source_documents']])
        urls_reg_list = []
        for count, url in enumerate(urls_answer):
                url_reg = regex_source(url)
                urls_reg_list.append(url_reg)
                print(str(count+1)+":", url_reg)
        url_list.append(urls_reg_list)

    #Create DataFrame
    table_dict = {"question" : question_list, "answer" : response_list, "inference_time" : time_list,
                  "Reference" : url_list }

    df = pd.DataFrame(table_dict)
    date_time = datetime.datetime.now().strftime("%d%b")
    filename = f"omnniscien-7a-finetune-merge-phrase-omniscien-with-rag.csv"
    df.to_csv(filename)
    print("finish")

if __name__ == '__main__':
        main()
