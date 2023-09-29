import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

import requests
#import captcha_solver #pip install captcha_solver
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


#scrap website
from bs4 import BeautifulSoup
import requests

#Langchain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator


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

def split_docs(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sp_docs = text_splitter.split_documents(documents)
    return sp_docs

def get_similiar_docs(db, query,k=1,score=False):
    if score:
      similar_docs = db.similarity_search_with_score(query,k=k)
    else:
      similar_docs = db.similarity_search(query,k=k)
    return similar_docs

def regex_source(answer):
    pattern = r"'source': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    return matchs

def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING (Langchain)")
    model_emb = "all-MiniLM-L6-v2"
    repo_id = "declare-lab/flan-alpaca-large"
    embedding_function = SentenceTransformerEmbeddings(model_name=model_emb)

    #documents = load_docs('omniscien.com')
    # sp_docs = split_docs(documents)
    # create the open-source embedding function
    # db = Chroma.from_documents(sp_docs, embedding_function, persist_directory="./chroma_db")
    # db.persist()
    # load it into Chroma
    db = Chroma(persist_directory="chroma_db_hugingface", embedding_function=embedding_function)

    llm_hug = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 1024})
    chain = load_qa_with_sources_chain(llm_hug,  chain_type="stuff")
    
    
    # index = VectorstoreIndexCreator(
    # text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0),
    # embedding = embedding_function, vectorstore_cls = Chroma).from_documents(documents)
    
    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
      #Document QA
      
      similar_docs = db.similarity_search(query)
      ans_qa = chain({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
      ans_qa_with_sc = chain({"input_documents": similar_docs, "question": query}, return_only_outputs=False)
      
      # #Retrieval QA
      rqa = RetrievalQA.from_chain_type(llm=llm_hug, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True)
      ans_rqa = rqa({"query": query})
      
      #Document QA
      st.markdown('Document QA')
      st.write(ans_qa["output_text"])
      st.write(regex_source(ans_qa_with_sc))

      # #Retrieval QA
      st.markdown('ref: Retrieval QA')
      st.write(ans_rqa["result"])
      st.write(regex_source(ans_rqa["source_documents"]))

      # # vector index
      # ans = index.query(llm=llm_hug, question=query, chain_type = "stuff")
      # st.write(ans)
      
    #Document QA <--- Load from local
    #Retrieval Qa <--- Load from local
    #vector index <-- create and load 
    
    

if __name__ == '__main__':
	main()
    
