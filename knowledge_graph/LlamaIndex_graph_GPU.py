import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from llama_index import(
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
    )

# upload model
from llama_index.llms import LangChainLLM
from llama_index.graph_stores import SimpleGraphStore
from llama_index import (KnowledgeGraphIndex)
from llama_index.storage.storage_context import StorageContext
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

url = "omniscien.com"
model_emb_path = "sentence-transformers/all-mpnet-base-v2"
documents = SimpleDirectoryReader(url, recursive = True).load_data()
print(len(documents))



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


llm = LangChainLLM(load_llm())
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1000)

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

if len(documents) > 160 :
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=2,
        storage_context=storage_context,
        service_context=service_context,
        show_progress = True
    )
    index.set_index_id("vector_index_graph")
    index.storage_context.persist("./llama_vector_graph")
    print("finish")
else :
    print("data not complete")