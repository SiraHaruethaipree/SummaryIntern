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
    Prompt
    )

from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

# upload model 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
import re
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
    llm = HuggingFaceHub(repo_id = model_path, model_kwargs = {"temperature":0, "max_length":512}) #770M parameters			
    return llm   

def custom_prompt():
    TEMPLATE_STR = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    """Given this information, please return only useful answer.
    Each response should consist of at least two sentences, with a minimum length requirement. 
    Avoid using redundant or repetitive phrases in your response.
    If you don't know the answer, please just say that you don't know the answer, 
    Don't try to make up an answer and we encourage you to present diverse sentence structures and formats in your answers, 
    rather than relying on the same patterns repeatedly for each sentence. : {query_str}\n""")
    QA_TEMPLATE = Prompt(TEMPLATE_STR)
    return QA_TEMPLATE

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
    pattern = r"'file_name': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    convert_urls = convert_to_website_format(matchs)
    res_urls = check_duplicate(source_list=convert_urls)
    return res_urls


def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES (LlamaIndex)")
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

    storage_context = StorageContext.from_defaults(persist_dir="./llama_vector_index_v2")
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine(
         streaming=False, 
         similarity_top_k=2, 
         service_context=service_context, 
         text_qa_template=custom_prompt())


    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        start = time.time()
        response_stream = query_engine.query(query)
        st.markdown('LLama Index k = 2')
        st.write(response_stream.response)
        urls = regex_source(response_stream.get_formatted_sources)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

      
if __name__ == '__main__':
	main()