from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

import time
import os
import re

def load_docs(docs_path):
    loader = DirectoryLoader(docs_path, glob="**/*.html")
    documents = loader.load()
    return documents

def clean_duplicate(documents):
    content_unique = []
    index_unique = []
    content_duplicate = []
    index_duplicate = []
    for index, doc in enumerate(documents):
        if doc.page_content not in content_unique:
            content_unique.append(doc.page_content)
            index_unique.append(index)
        else :
            content_duplicate.append(doc.page_content)
            index_duplicate.append(index)
    documents_clean = [item for index, item in enumerate(documents) if index in index_unique]
    return documents_clean

def split_docs(documents,chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    sp_docs = text_splitter.split_documents(documents)
    return sp_docs

def check_page_not_found(documents):
    doc_found = []
    index_found = []
    for index, doc in enumerate(documents):
        if "The page you were looking for does not exist." not in str(doc.page_content):
            doc_found.append(doc.page_content)
            index_found.append(index)
    documents_found = [item for index, item in enumerate(documents) if index in index_found]
    return documents_found

def main():
    documents = load_docs('omniscien.com')
    documents_found = check_page_not_found(documents)
    documents_clean = clean_duplicate(documents_found)
    sp_docs = split_docs(documents_clean)
    DB_FAISS_PATH = "vectorstores_clean_doc_gte-base_no_overlap/db_faiss"
    embedding_function = HuggingFaceEmbeddings(model_name = "thenlper/gte-base",
                            model_kwargs = {'device': 'cuda'})
    db = FAISS.from_documents(sp_docs, embedding_function)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
	main()