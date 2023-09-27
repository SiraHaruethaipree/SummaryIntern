import streamlit as st
import os
import re 
import time

from llama_index import(
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
    LangchainEmbedding,
    VectorStoreIndex,
    load_index_from_storage,
    load_graph_from_storage,
    LLMPredictor,
    PromptHelper,
    get_response_synthesizer,
    QueryBundle,
    Prompt,
    )

# upload model
from llama_index.llms import LangChainLLM
from llama_index.graph_stores import SimpleGraphStore
from llama_index import (KnowledgeGraphIndex)
from llama_index.storage.storage_context import StorageContext
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.query_engine import RetrieverQueryEngine

# import NodeWithScore
from llama_index.schema import NodeWithScore
# Retrievers
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
from typing import List

# import sys
# sys.setrecursionlimit(10000) # 10000 is an example, try with different values

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

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

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

@st.cache_resource 
def load_llm():
    n_gpu_layers = 32 
    n_batch = 512  
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
        callback_manager=callback_manager,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=True,
        n_ctx = 4096, 
        temperature = 0.1, 
        max_tokens = 4096
    )
    return llm

def main():
    global index
    st.header("DOCUMENT QUESTION ANSWERING FOR OMNISCIEN TECHNOLOGIES (LlamaIndex)")
    llm_predictor = LLMPredictor(llm=LangChainLLM(llm = load_llm()))
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs = {'device': 'cuda'}))
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            chunk_size=1000, 
            embed_model = embed_model)
    #Graph Vector
    storage_context_graph = StorageContext.from_defaults(persist_dir="./llama7b_graph_index_removeHTML_MaxTriplets10")
    kg_index = load_index_from_storage(storage_context = storage_context_graph, service_context=service_context)
    #Index vector
    storage_context_vector = StorageContext.from_defaults(persist_dir="./llama7b_vector_index_removeHTML")
    vector_index = load_index_from_storage(storage_context = storage_context_vector, service_context=service_context)
    #create custom retriever
    vector_retriever = VectorIndexRetriever(index=vector_index)
    kg_retriever = KGTableRetriever(
        index=kg_index, retriever_mode="keyword", include_text=True
    )
    custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",
        streaming = False,
        text_qa_template=custom_prompt())
    
    custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer)

    vector_query_engine = vector_index.as_query_engine(
        streaming=False,
        service_context=service_context,
        similarity_top_k=5,
        text_qa_template=custom_prompt())

    kg_keyword_query_engine = kg_index.as_query_engine(
        # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
        include_text=False,
        retriever_mode="keyword",
        response_mode="tree_summarize",
        text_qa_template=custom_prompt())
    
    query = st.text_input("ASK ABOUT THE DOCS (Llama7b):")	
    if query:
        start = time.time()
        response_stream = kg_keyword_query_engine.query(query)
        st.markdown('KG engine')
        st.write(response_stream.response)
        urls = regex_source(response_stream.get_formatted_sources)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

        start = time.time()
        response_stream = vector_query_engine.query(query)
        st.markdown('Vector engine')
        st.write(response_stream.response)
        urls = regex_source(response_stream.get_formatted_sources)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")

        start = time.time()
        response_stream = custom_query_engine.query(query)
        st.markdown('Custom engine')
        st.write(response_stream.response)
        urls = regex_source(response_stream.get_formatted_sources)
        for count, url in enumerate(urls):
             st.write(str(count+1)+":", url)
        end = time.time()
        st.write("Respone time:",int(end-start),"sec")


if __name__ == '__main__':
	main()
