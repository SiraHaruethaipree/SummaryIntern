import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import re


#huggingface
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import time

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
    llm = CTransformers(model = model_path,
                        model_type = "llama",
                        max_new_tokens = 512,
                        temperature = 0.1)
    return llm

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context : {context}
Question : {question}

The answer should consist of at least 3 sentences. Only returns the helpful and reasonable answer $
No need to return the question.I just want answer.
Helpful answer:
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context',
                                                                              'question'])
    return prompt

def load_llm():
    llm = CTransformers(model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type = "llama",
                        max_new_tokens = 512,
                        temperature = 0.1)
    return llm


# def qa_bot():
#     DB_FAISS_PATH = "vectorstores/db_faiss"
#     embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs = {'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)
#     return qa

# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response

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
    urls_remove = ["www.omniscien.com/aboutus/company"]
    # Remove the URL from the list
    filtered_urls = [url for url in urls if url not in  urls_remove]
    return filtered_urls


def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES (Langchain FAISS Meta-Llama2) K = 2")
    DB_FAISS_PATH = "vectorstores/db_faiss"
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt(custom_prompt_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {'k':2}), 
        return_source_documents = True,
        chain_type_kwargs = {"prompt":qa_prompt})


    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        start = time.time()
        response = qa_chain({'query': query})
        st.write(response["result"])
        urls = regex_source(response)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

      
if __name__ == '__main__':
	main()