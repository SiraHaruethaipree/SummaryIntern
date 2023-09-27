from transformers import pipeline
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
    
def clean_duplicate(documents):
    content_unique = []
    index_unique = []
    content_duplicate = []
    index_duplicate = []
    for index, doc in enumerate(documents):
        if doc.text not in content_unique:
            content_unique.append(doc.text)
            index_unique.append(index)
        else :
            content_duplicate.append(doc.text)
            index_duplicate.append(index)
    documents_clean = [item for index, item in enumerate(documents) if index in index_unique]
    return documents_clean

def extract_triplets(input_text):
    triplet_extractor = pipeline('text2text-generation', 
                            model='Babelscape/rebel-large', 
                            tokenizer='Babelscape/rebel-large', 
                            device='cuda')
    text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])[0]

    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets

def load_llm():
    n_gpu_layers = 32 
    n_batch = 512  
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="/home/sira/sira_project/meta-Llama2/llama-2-7b-chat.ggmlv3.q8_0.bin",
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
    documents_clean = clean_duplicate(documents)

    llm_predictor = LLMPredictor(llm=LangChainLLM(llm = load_llm()))
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name = "thenlper/gte-base",
            model_kwargs = {'device': 'cpu'}))
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            chunk_size = 256, 
            embed_model = embed_model)
    index = KnowledgeGraphIndex.from_documents(documents_clean, 
                                               kg_triplet_extract_fn=extract_triplets, 
                                               service_context=service_context,
                                               show_progress = True)
    
    index.set_index_id("vector_index_graph")
    index.storage_context.persist("./llama7b_graph_index_removeHTML_Rebel")
    print("finish")

if __name__ == '__main__':
	main()
