import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


#huggingface
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#LLama_index
from llama_index import(
    GPTVectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    LangchainEmbedding,
    StorageContext,
    load_index_from_storage,
    )

from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

# upload model 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub

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

def load_llm(model_path):      
    llm = HuggingFaceHub(repo_id = model_path, model_kwargs = {"temperature":0, "max_length":512}) #770M parameters			
    return llm   

def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING OMNISCIEN")
    model_path = "declare-lab/flan-alpaca-large"
    model_emb_path = "sentence-transformers/all-mpnet-base-v2"

    llm = load_llm(model_path)
    llm_predictor = LLMPredictor(llm = llm)
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_emb_path))

    max_input_size = 4096
    num_output = 512
    max_chunk_overlap = 0.20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embed_model,
    prompt_helper=prompt_helper)

    storage_context = StorageContext.from_defaults(persist_dir="./llama_index_docs_api_v1")
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine(
         streaming=False, similarity_top_k=1, service_context=service_context)


    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        response_stream = query_engine.query(query)
        st.markdown('LLama Index')
        st.write(response_stream.response)
      
    
    
    

if __name__ == '__main__':
	main()