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
    Prompt
    )

# TO UPLOAD LOCAL LLMs
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
            #url = 'www.' + url
            url = "https://" + url
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

def load_llm():
    n_gpu_layers = 32 
    n_batch = 512  
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        #model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_path="llama-2-13b-chat.ggmlv3.q4_0.bin",
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
    llm_predictor = LLMPredictor(llm=LangChainLLM(llm = load_llm()))
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name = "thenlper/gte-base",
            #model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs = {'device': 'cuda'}))
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            chunk_size=1000, 
            embed_model = embed_model)
    #Index vector
    storage_context_vector = StorageContext.from_defaults(persist_dir="./website_docs_index_by_llamaIndexRay")
    vector_index = load_index_from_storage(storage_context = storage_context_vector, service_context=service_context)

    #Graph vector
    storage_context_graph = StorageContext.from_defaults(persist_dir="./llama7b_graph_index_removeHTML")
    kg_index = load_index_from_storage(storage_context = storage_context_graph, service_context=service_context)
    
    #create custom retriever
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    kg_retriever = KGTableRetriever(
        index=kg_index, retriever_mode="keyword", include_text=True
    )
    custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        response_mode="tree_summarize",
        streaming = False,)
        #text_qa_template=custom_prompt())
    
    custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer)

    vector_query_engine = vector_index.as_query_engine(
    streaming=False,
    service_context=service_context,
    similarity_top_k=5,
    text_qa_template=custom_prompt()
    )


    query = input("ASK ABOUT THE DOCS:")
    #start = time.time()
    response_stream = vector_query_engine.query(query)
    print(str(response_stream))

    # response_stream = custom_query_engine.query(query)
    # print(response_stream.response)
    # print(str(response_stream.get_formatted_sources))
    #urls = regex_source(response_stream.get_formatted_sources)
    # for count, url in enumerate(urls):
    #         st.write(str(count+1)+":", url)
    # end = time.time()
    # st.write("Respone time:",int(end-start),"sec")

if __name__ == '__main__':
	main()