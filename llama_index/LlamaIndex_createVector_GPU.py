import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from llama_index import(
    GPTVectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    LangchainEmbedding,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    VectorStoreIndex
    )

from langchain.document_loaders import DirectoryLoader
from langchain.docstore.document import Document as LangchainDocument
from llama_index.node_parser import SimpleNodeParser

# upload model 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from llama_index import download_loader 

def load_llm(model_path):      
    llm = HuggingFaceHub(repo_id = model_path, model_kwargs = {"temperature":0, "max_length":1024}) #770M parameters			
    return llm   

url = "omniscien.com"
model_path = "declare-lab/flan-alpaca-large"
model_emb_path = "sentence-transformers/all-mpnet-base-v2"
filename_fn = lambda filename: {'file_name': filename}
documents = SimpleDirectoryReader(url, recursive = True, file_metadata=filename_fn).load_data()
print(len(documents))
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)


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
index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
index.set_index_id("vector_index")
index.storage_context.persist("./llama_vector_index_v2")
print("finish")