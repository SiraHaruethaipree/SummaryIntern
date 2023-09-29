from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from IPython.display import Markdown, display
from llama_index.node_parser import SimpleNodeParser
import chromadb

# define embedding function
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda:2"})
)

# load documents
url = "omniscien.com"
filename_fn = lambda filename: {'file_name': filename}
documents = SimpleDirectoryReader(url, recursive = True, file_metadata=filename_fn).load_data()

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

chroma_client = chromadb.PersistentClient(path="chroma_db_llamaindex")
chroma_collection = chroma_client.create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context, show_progress=True)
print("finish")