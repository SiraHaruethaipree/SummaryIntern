import os
from urllib import response
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pickle
import re

import requests
#import captcha_solver #pip install captcha_solver
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# new import packages
import tiktoken

# import streamlit as st
# import pinecone
import json


import tiktoken
import hashlib

#scrap website
from bs4 import BeautifulSoup
import requests



#for huggingface
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#Langchain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

#for Huggingface
from langchain import HuggingFaceHub

#for openai
from langchain.llms import OpenAI
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# zzh
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle
#pip install streamlit

index = None
url = ""

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
	
	add_vertical_space(3)
	st.write ('Made this app for testing Document Question Answering with Custom URL Data')


def load_docs(docs_path):
    loader = DirectoryLoader(docs_path, glob="**/*.html")
    documents = loader.load()
    return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sp_docs = text_splitter.split_documents(documents)
    return sp_docs

# def get_similiar_docs(db, query,k=1,score=False):
#     if score:
#       similar_docs = db.similarity_search_with_score(query,k=k)
#     else:
#       similar_docs = db.similarity_search(query,k=k)
#     return similar_docs

def regex_source(answer):
    pattern = r"'source': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    return matchs

def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING")
    model_emb = "all-MiniLM-L6-v2"
    repo_id = "declare-lab/flan-alpaca-large"

    #documents = load_docs('omnisciencom')
    #sp_docs = split_docs(documents)
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=model_emb)
    # load it into Chroma
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    llm_hug = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 768})
    chain = load_qa_with_sources_chain(llm_hug,  chain_type="stuff")

    
    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
      #Document QA
      similar_docs = db.similarity_search(query)
      ans_qa = chain({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
      ans_qa_with_sc = chain({"input_documents": similar_docs, "question": query}, return_only_outputs=False)
      
      #Retrieval QA
      rqa = RetrievalQA.from_chain_type(llm=llm_hug, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True)
      ans_rqa = rqa({"query": query})
      
      #Document QA
      st.markdown('ref: Document QA')
      st.write(ans_qa["output_text"])
      st.write(regex_source(ans_qa_with_sc))

      #Retrieval QA
      st.markdown('ref: Retrieval QA')
      st.write(ans_rqa["result"])
      st.write(regex_source(ans_rqa["source_documents"]))
      
    
    
    

if __name__ == '__main__':
	main()