import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


#huggingface
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
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
    llm = CTransformers(model = model_path,
                        model_type = "llama",
                        max_new_tokens = 512,
                        temperature = 0.1)
    return llm

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context : {summaries}
Question : {question}

Only returns the helpful answer below and nothing else.
Each response should consist of at least two sentences, with a minimum length requirement. 
Avoid using redundant or repetitive phrases in your response.
Don't try to make up an answer and we encourage you to present diverse sentence structures and formats in your answers, 
rather than relying on the same patterns repeatedly for each sentence.
Helpful answer:
"""

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


def get_similiar_docs(db, query,k=2,score=False):
  if score:
    similar_docs = db.similarity_search_with_score(query,k=k)
  else:
    similar_docs = db.similarity_search(query,k=k)
  return similar_docs


def main():
    
    global index
    st.header("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES (Langchain Meta-Llama2)")
    model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['summaries','question'])
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="/home/sira/sira_project/DQA_demo/chroma_db_hugingface", embedding_function=embedding_function)
    chain = load_qa_with_sources_chain(llm = load_llm(model_path),  chain_type="stuff", prompt=prompt)


    query = st.text_input("ASK ABOUT THE DOCS:")	
    if query:
        start = time.time()
        similar_docs = get_similiar_docs(db, query,k=2)
        #st.write(similar_docs)
        ans_qa = chain({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
        #st.markdown('LLama Index')
        st.write(ans_qa["output_text"])
        # urls = regex_source(ans_qa["source_documents"])
        # for count, url in enumerate(urls):
        #      st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

      
if __name__ == '__main__':
	main()