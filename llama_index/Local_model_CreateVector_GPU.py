import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from llama_index import(
    GPTVectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    Document,
    VectorStoreIndex,
    LangchainEmbedding,
    StorageContext,
    load_index_from_storage,
    )

from llama_index import SimpleDirectoryReader
from langchain import OpenAI
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

# upload model 
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from llama_index.llms import LangChainLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

url = "omniscien.com"
model_path = "orca-mini-3b.ggmlv3.q4_0.bin"
model_emb_path = "sentence-transformers/all-mpnet-base-v2"
def load_llm(model_path):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm_langchain = LlamaCpp(
    model_path= model_path, 
    callback_manager=callback_manager, 
    verbose=True, 
    n_ctx=2048) #define n-ctx for prevent exceed token error
    llm = LangChainLLM(llm=llm_langchain)
    return llm

llm = load_llm(model_path)
llm_predictor = LLMPredictor(llm = llm)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_emb_path))

max_input_size = 4096
num_output = 512
max_chunk_overlap = 0.20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                              embed_model=embed_model,
                                              prompt_helper=prompt_helper,
                                              chunk_size=1000, chunk_overlap=200)

documents = SimpleDirectoryReader(url, recursive = True).load_data()
print(len(documents))
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)


index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
index.set_index_id("Local_vector_index")
index.storage_context.persist("./orca_index_v1")
print("finish")