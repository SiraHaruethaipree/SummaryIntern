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
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
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


custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context : {summaries}
Question : {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""


def load_llm():
    llm = CTransformers(model = "/home/sira/sira_project/meta-Llama2/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type = "llama",
                        max_new_tokens = 512,
                        temperature = 0.5)
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
        convert_urls.append(url)
    return convert_urls

def regex_source(answer):
    pattern = r"'source': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    convert_urls = convert_to_website_format(matchs)
    res_urls = check_duplicate(source_list=convert_urls)
    return res_urls



def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES (Langchain FAISS Meta-Llama2) K = 2")
    st.subheader("Load_qa_chain")
    DB_FAISS_PATH = "vectorstores/db_faiss"
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['summaries',
                                                                              'question'])
    qa_chain = load_qa_with_sources_chain(load_llm(), chain_type="stuff", prompt=prompt)


    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        start = time.time()
        docs = db.similarity_search(query, k = 1)
        response = qa_chain({"input_documents": docs, "question": query}, return_only_outputs=False)
        st.write(response["output_text"])
        urls = regex_source(response)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

      
if __name__ == '__main__':
	main()