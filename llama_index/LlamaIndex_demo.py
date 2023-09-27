from tkinter import *
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

def load_llm(model_path):      
    llm = HuggingFaceHub(repo_id = model_path, model_kwargs = {"temperature":0, "max_length":512}) #770M parameters			
    return llm   


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

storage_context = StorageContext.from_defaults(persist_dir="./llama_vector_index")
index = load_index_from_storage(storage_context, service_context=service_context)
query_engine = index.as_query_engine(
        streaming=False, similarity_top_k=1, service_context=service_context)


window=Tk()
window.title('Document Question Answering Omniscien (LlamaIndex)')
window.geometry("300x200+10+20")
window.mainloop()
query=Entry()
query.place(x=60, y=50)

btn=Button(window, text="Answers", fg='blue')
btn.place(x=80, y=100)

	
if btn == True:
    response_stream = query_engine.query(query)
    #st.markdown('LLama Index')
    label=Label(response_stream.response, text="", font=('Calibri 15'))
    #st.write(response_stream.response)

# window=Tk()
# add widgets here


# label=Label(win, text="", font=('Calibri 15'))
# label.pack()

# btn=Button(window, text="Answers", fg='blue')
# btn.place(x=80, y=100)

# t2=Entry()
# t2.place(x=60, y=50)

window.title('Document Question Answering Omniscien (LlamaIndex)')
window.geometry("300x200+10+20")
window.mainloop()

# print(txtfld)