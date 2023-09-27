from langchain.document_loaders import UnstructuredURLLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate , Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import pinecone
import json

from llama_index import(
    GPTSimpleVectorIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    Document
)

from langchain import OpenAI
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

import tiktoken
import hashlib

from bs4 import BeautifulSoup
import requests


tokenizer = tiktoken.get_encoding("cli100K_base")

pinecone.init(
    api_key = "",
    environment = ""
)

OPRN_API_KEY = "..."

document = []

def scrape(site):
    urls = []
    
    def scrape_helper(current_site):
        nonlocal urls

        r = requests.get(current_site)

        s = BeautifulSoup(r.text, "html.parser")
        print(s.find_all("a"))
        for i in s.find_all("a"):
            if "href" in i.attrs:
                href = i.attrs["href"]

                if href.startswith("/") or href.startswith('#'):
                    full_url = site + href

                    if full_url not in urls:
                        urls.append(full_url)
                        scrape_helper(full_url)
    scrape_helper(site)
    return urls

def load_documnet_to_gpt_vectorstore(url):
    from llama_index import download_loader 

    urls = scrape(url)
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls)
    parser = SimpleNodeParser()

    nodes = parser.get_nodes_from_documents(documents)
    llm_predictor = LLMPredictor(
        llm = OpenAI(temperature=0, model_name = "text-davinci-003", openai_api_key = OPRN_API_KEY)
    )

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index = GPTSimpleVectorIndex(nodes, service_context=service_context)
    index.save_to_disk("./gpt_index_docs_api_remotion_v2.json")
    return index

def chat(query):
    index = GPTSimpleVectorIndex.load_from_disk("gpt_index_docs.json")
    response = index.query(query)
    print(response)
    return response

st.header("Docs")

doc_input = st.text_input("paste documentation url")

if st.button("load documnets"):
    st.markdown(load_documnet_to_gpt_vectorstore(doc_input))

user_input = st.text_input("ask about the docs")
if st.button("Ask"):
    st.markdown(chat(user_input))