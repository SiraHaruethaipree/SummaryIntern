from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

import time
import os
import re
import streamlit as st

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
	st.write ('Made this app for testing Document Question Answering with Custom URL Data')

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer. 

Context : {context}
Question : {question}

The answer should consist of at least 7 sentences. Only returns the helpful and reasonable answer below and nothing else.
No need to return the question.I just want answer.
Helpful answer:
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context',
                                                                              'question'])
    return prompt

def load_docs(docs_path):
    loader = DirectoryLoader(docs_path, glob="**/*.html")
    documents = loader.load()
    return documents

def clean_duplicate(documents):
    content_unique = []
    index_unique = []
    content_duplicate = []
    index_duplicate = []
    for index, doc in enumerate(documents):
        if doc.page_content not in content_unique:
            content_unique.append(doc.page_content)
            index_unique.append(index)
        else :
            content_duplicate.append(doc.page_content)
            index_duplicate.append(index)
    documents_clean = [item for index, item in enumerate(documents) if index in index_unique]
    return documents_clean

def split_docs(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sp_docs = text_splitter.split_documents(documents)
    return sp_docs

def load_llm():
    n_gpu_layers = 32  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,n_ctx = 4096, temperature = 0.1, max_tokens = 4096
    )
    return llm

def check_duplicate(source_list):
    res = []
    for i in source_list:
        if i not in res:
            res.append(i)
    return res

def convert_to_website_format(urls):
    convert_urls = []
    for url in urls:
        # Remove any '.html' at the end of the URL
        url = re.sub(r'\.html$', '', url)
        # Check if the URL starts with 'www.' or 'http://'
        if not re.match(r'(www\.|http://)', url):
            url = 'www.' + url
        if '/index' in url:
            url = url.split('/index')[0]
        match = re.match(r'^([^ ]+)', url)
        if match:
            url = match.group(1)
        convert_urls.append(url)
    return convert_urls

def regex_source(answer):
    pattern = r"'source': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    convert_urls = convert_to_website_format(matchs)
    res_urls = check_duplicate(source_list=convert_urls)
    res_urls = filter_similar_url(res_urls)
    return res_urls

def filter_similar_url(urls):
    urls_remove = ["www.omniscien.com/aboutus/company","www.omniscien.com/lsev6/asr/automatic-speech-recognition-overview", "www.omniscien.com/lsev6/features/asr/autonomous-speech-recognition-overview","www.omniscien.com/lsev6/asr"]
    # Remove the URL from the list
    filtered_urls = [url for url in urls if url not in  urls_remove]
    return filtered_urls

def main():
    global index
    st.header("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES")
    st.subheader("Retrival QA (Running with GPU, max_token = 4096, K=5)")
    st.subheader("Test Embedding")
    documents = load_docs('omniscien.com')
    documents_clean = clean_duplicate(documents)
    sp_docs = split_docs(documents_clean)
    qa_prompt = set_custom_prompt(custom_prompt_template)
    # embeddings_MiniLM = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
    #                             model_kwargs = {'device': 'cuda'})
    # embeddings_bge_small_en = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en",
    #                             model_kwargs = {'device': 'cuda'})
    # embeddings_gte_base = HuggingFaceEmbeddings(model_name = "thenlper/gte-base",
    #                             model_kwargs = {'device': 'cuda'})
    # embeddings_gte_small = HuggingFaceEmbeddings(model_name = "thenlper/gte-small",
    #                             model_kwargs = {'device': 'cuda'})
    embeddings_e5_base = HuggingFaceEmbeddings(model_name = "intfloat/e5-base",
                                model_kwargs = {'device': 'cuda'})
    # db_MiniLM = FAISS.from_documents(sp_docs, embeddings_MiniLM)
    # db_bge_small_en = FAISS.from_documents(sp_docs, embeddings_bge_small_en)
    # db_gte_base = FAISS.from_documents(sp_docs, embeddings_gte_base)
    # db_gte_small = FAISS.from_documents(sp_docs, embeddings_gte_small)
    db_e5_base = FAISS.from_documents(sp_docs, embeddings_e5_base)
    llm = load_llm()
    # qa_prompt = set_custom_prompt(custom_prompt_template)
    # qa_chain_MiniLM = RetrievalQA.from_chain_type(
    #     llm = llm,
    #     chain_type = "stuff",
    #     retriever = db_MiniLM.as_retriever(search_kwargs = {'k':5}), 
    #     return_source_documents = True,
    #     chain_type_kwargs = {"prompt":qa_prompt}) 
    
    # qa_chain_small_en = RetrievalQA.from_chain_type(
    #     llm = llm,
    #     chain_type = "stuff",
    #     retriever = db_bge_small_en.as_retriever(search_kwargs = {'k':5}), 
    #     return_source_documents = True,
    #     chain_type_kwargs = {"prompt":qa_prompt}) 

    # qa_chain_gte_base = RetrievalQA.from_chain_type(
    #     llm = llm,
    #     chain_type = "stuff",
    #     retriever = db_gte_base.as_retriever(search_kwargs = {'k':5}), 
    #     return_source_documents = True,
    #     chain_type_kwargs = {"prompt":qa_prompt}) 

    # qa_chain_gte_small = RetrievalQA.from_chain_type(
    #     llm = llm,
    #     chain_type = "stuff",
    #     retriever = db_gte_small.as_retriever(search_kwargs = {'k':5}), 
    #     return_source_documents = True,
    #     chain_type_kwargs = {"prompt":qa_prompt}) 

    qa_chain_e5_base = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db_e5_base.as_retriever(search_kwargs = {'k':5}), 
        return_source_documents = True,
        chain_type_kwargs = {"prompt":qa_prompt}) 
    
    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        # st.markdown('all-MiniLM-L6-v2(Current)')
        # start = time.time()
        # response = qa_chain_MiniLM({'query': query})
        # st.write(response["result"])
        # urls = regex_source(response)
        # for count, url in enumerate(urls):
        #      st.write(str(count+1)+":", url)
        # end = time.time()
        # st.write("Respone time:",int(end-start),"sec")

        # st.markdown('BAAI/bge-small-en')
        # start = time.time()
        # response = qa_chain_small_en({'query': query})
        # st.write(response["result"])
        # urls = regex_source(response)
        # for count, url in enumerate(urls):
        #      st.write(str(count+1)+":", url)
        # end = time.time()
        # st.write("Respone time:",int(end-start),"sec")

        # st.markdown('thenlper/gte-base')
        # start = time.time()
        # response = qa_chain_gte_base({'query': query})
        # st.write(response["result"])
        # urls = regex_source(response)
        # for count, url in enumerate(urls):
        #      st.write(str(count+1)+":", url)
        # end = time.time()
        # st.write("Respone time:",int(end-start),"sec")

        # st.markdown('thenlper/gte-small')
        # start = time.time()
        # response = qa_chain_gte_small({'query': query})
        # st.write(response["result"])
        # urls = regex_source(response)
        # for count, url in enumerate(urls):
        #      st.write(str(count+1)+":", url)
        # end = time.time()
        # st.write("Respone time:",int(end-start),"sec")

        st.markdown('intfloat/e5-base')
        start = time.time()
        response = qa_chain_e5_base({'query': query})
        st.write(response["result"])
        urls = regex_source(response)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

if __name__ == '__main__':
	main()
