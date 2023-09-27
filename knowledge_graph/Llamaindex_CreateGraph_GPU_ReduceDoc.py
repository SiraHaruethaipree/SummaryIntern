from llama_index import(
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
    LangchainEmbedding,
    VectorStoreIndex,
    LLMPredictor,
    KnowledgeGraphIndex
    ) 
    
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import LangChainLLM
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import SimpleGraphStore

from typing import Callable, Dict, Generator, List, Optional, Type
from pathlib import Path
from llama_index.readers.base import BaseReader
from llama_index.schema import Document
import logging


logger = logging.getLogger(__name__)
class DirectorySearchSource():
    def __init__(
    self,
    num_files_limit: Optional[int] = None,
    exclude_hidden: bool = True,
    required_exts: Optional[List[str]]  = None,
    recursive : bool = True,):
    
        super().__init__()

        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit

    def add_files(self, input_dir):
        all_files = set()
        rejected_files = set()
        list_files = []

        file_refs: Generator[Path, None, None]
        if self.recursive:
            file_refs = Path(input_dir).rglob("*")
        else:
            file_refs = Path(input_dir).glob("*")
        for ref in file_refs:
            # Manually check if file is hidden or directory instead of
            # in glob for backwards compatibility.
            is_dir = ref.is_dir()
            skip_because_hidden = self.exclude_hidden and ref.name.startswith(".")
            skip_because_bad_ext = (
                self.required_exts is not None and ref.suffix not in self.required_exts
            )
            skip_because_excluded = ref in rejected_files

            if (
                is_dir
                or skip_because_hidden
                or skip_because_bad_ext
                or skip_because_excluded
            ):
                continue
            else:
                all_files.add(ref)
        new_input_files = sorted(list(all_files))

        if len(new_input_files) == 0:
            raise ValueError(f"No files found in {input_dir}.")

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        logger.debug(
            f"> [SimpleDirectoryReader] Total files added: {len(new_input_files)}")

        for f in new_input_files:
            list_files.append(str(f))
        return list_files

class HtmlFilesReader(BaseReader):
    """Simple web page reader.

    Reads pages from the web.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.

    """

    def __init__(self, html_to_text: bool = False):
        """Initialize with parameters."""
        try:
            import html2text  # noqa: F401
        except ImportError:
            raise ImportError(
                "`html2text` package not found, please run `pip install html2text`"
            )
        self._html_to_text = html_to_text

    def load_data(self, input_files, ):
        """Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(input_files, list):
            raise ValueError("input_files must be a list of strings.")
        documents = []
        for input_file in input_files:
            #response = requests.get(url, headers=None).text
            with open(input_file, "r", errors = "ignore", encoding='utf-8') as f:
                response = f.read()
            if self._html_to_text:
                import html2text

                response = html2text.html2text(response)

            doc = Document(text=response)
            doc.metadata = {'file_name': input_file}


            documents.append(doc)

        return documents

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
    input_dir = "./omniscien.com"
    lists_files = DirectorySearchSource().add_files(input_dir)
    documents = HtmlFilesReader(html_to_text=True).load_data(input_files = lists_files)
    llm_predictor = LLMPredictor(llm=LangChainLLM(llm = load_llm()))
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs = {'device': 'cuda'}))
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            chunk_size=1000, 
            embed_model = embed_model)

    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # NOTE: can take a while!
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=3,
        storage_context=storage_context,
        service_context=service_context,
        show_progress = True
    )

    index.set_index_id("vector_index_graph")
    index.storage_context.persist("./llama7b_vector_index_removeHTML")
    print("finish")

if __name__ == '__main__':
	main()