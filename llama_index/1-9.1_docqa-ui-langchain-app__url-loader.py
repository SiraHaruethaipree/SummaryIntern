import os
from urllib import response
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import pickle

import requests
#import captcha_solver #pip install captcha_solver
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# new import packages
from langchain.document_loaders import UnstructuredURLLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate , Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
# import streamlit as st
# import pinecone
import json

from llama_index import(
    GPTVectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    Document,
    VectorStoreIndex,
    LangchainEmbedding,
    StorageContext,
    load_index_from_storage,
    )

from langchain import OpenAI
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

import tiktoken
import hashlib

#scrap website
from bs4 import BeautifulSoup
import requests

#upload model 
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from llama_index.llms import LangChainLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


#for huggingface
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# zzh
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
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


def scrape_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        urls = []
        # Customize this part based on the structure of the website
        # Here's an example of extracting URLs from <a> tags
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and href.startswith('http'):
                urls.append(href)
        return urls
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return []
    
def load_llm(model_path):    
    		
    #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    #CTX_MAX = 8192
    #llm_langchain = LlamaCpp(
    #model_path= model_path, callback_manager=callback_manager, verbose=False, n_ctx=CTX_MAX)
    #llm = LangChainLLM(llm=llm_langchain) 
    
    
    repo_id = model_path
    llm = HuggingFaceHub(repo_id = repo_id, model_kwargs = {"temperature":0, "max_length":512}) #770M parameters	
			
    return llm   


def load_document_to_gpt_vectorstore(url, model_path, model_emb_path):
    from llama_index import download_loader 

    urls = scrape_url(url)
    #print("SCRAPTED URLs:\t", urls)
    
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls)
    parser = SimpleNodeParser()

    nodes = parser.get_nodes_from_documents(documents)
    #print("CONVERTED NODEs:\t", nodes)

    llm = load_llm(model_path)
    llm_predictor = LLMPredictor(llm = llm)
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_emb_path))
    #print("FINISHED EMBEDDINGS!")

    max_input_size = 4096
    num_output = 512
    max_chunk_overlap = 0.20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embed_model,
    prompt_helper=prompt_helper,
    )

    index = GPTVectorStoreIndex(nodes, service_context=service_context)
    index.storage_context.persist(persist_dir="./llama_index_for_ui")
    #print("SAVED TO DISK FOR DOCS JSON!")
    
    return index, service_context


def chat(query, index):
    #index = VectorStoreIndex.load_from_disk("./llama_index_for_ui")
    #response = index.query(query)
	response = index.as_query_engine().query(query)
    #print(response)
	return response


def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING")
    
    url = st.text_input("ENTER THE URL:")
    
    if url:
            urls_lst = scrape_url(url)
            st.write(urls_lst)
            
            model_path = "declare-lab/flan-alpaca-large"
            #model_path = "./ggml_model_q4_0.bin"
            model_emb_path = "sentence-transformers/all-mpnet-base-v2"
            #st.markdown(load_document_to_gpt_vectorstore(url= url, model_path= model_path,model_emb_path=model_emb_path) )
            
            #model_path = "declare-lab/flan-alpaca-large"
            #model_emb_path = "sentence-transformers/all-mpnet-base-v2"
            index, service_context = load_document_to_gpt_vectorstore(url= url, 
																model_path= model_path, 
                                                                model_emb_path=model_emb_path) 
            
    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        from llama_index import load_index_from_storage
        storage_context = StorageContext.from_defaults(persist_dir="./llama_index_for_ui")
        index = load_index_from_storage(storage_context, service_context=service_context)
        
        query_engine = index.as_query_engine(streaming=False, similarity_top_k=1, service_context=service_context)
        response_stream = query_engine.query(query)
        #st.write(response_stream.print_response_stream())
        
        #response = chat(query, index)
        st.write(response_stream.response)

            
                 
    #query = st.text_input("ASK ABOUT THE DOCS:")	
    #if st.button("ASK"):
        #if query:
            # Check if index is not None before using it in the "ASK" button click function
            #if index is not None:
                #response = chat(query, index)
                #st.write(response)
            #else:
                 #st.write("None")
              
    #response = index.as_query_engine().query(query)
    #st.write(response) 
	

    
    
    
    

if __name__ == '__main__':
	main()
	
