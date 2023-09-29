import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
import tkinter as tk
from tkinter import filedialog
from PyPDF2 import PdfReader
import os
import pathlib
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import DataFrameLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import JSONLoader
import re

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def check_collect_all_branch(web):#function for scrap whole website
    pattern = r'^all-https?://'
    if re.match(pattern, web):
        return True
    else:
        return False

def remove_prefix(web): #function for scrap whole website
    if web.startswith("All-"):
        return web[len("All-"):]
    elif web.startswith("all-"):
        return web[len("all-"):]
    else:
        return web

def get_all_branch_urls(root_url): #function for scrap whole website
    branch_urls = []

    response = requests.get(root_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        links = soup.find_all('a')

        for link in links:
            href = link.get('href')

            if href and not urlparse(href).netloc:
                absolute_url = urljoin(root_url, href)

                if absolute_url not in branch_urls:
                    branch_urls.append(absolute_url)
                    
    branch_urls = [link for link in branch_urls if link.startswith("https://")]
    return branch_urls

def read_url(web): # comment function collect whole website now it not work.
    # status = check_collect_all_branch(web.lower()) 
    # st.write(status)
    # if status == True :
    #     root_url = remove_prefix(web)
    #     st.write(f"remove prefix {root_url}")
    #     branch_urls = get_all_branch_urls(root_url)
    #     st.write(f"This website contain: {branch_urls} pages")
    #     loader = UnstructuredURLLoader(urls=branch_urls)
    #     data = loader.load()
    #     return data
    # else :
    #     url = [web]
    #     loader = UnstructuredURLLoader(urls=url)
    #     data = loader.load()
    #     return data
    url = [web]
    loader = UnstructuredURLLoader(urls=url)
    data = loader.load()
    return data

class UploadDoc:
    def __init__(self, path_data):
        self.path_data = path_data

    def prepare_filetype(self):
        extension_lists = {
            ".docx": [],
            ".pdf": [],
            ".html": [],
            ".png": [],
            ".xlsx": [],
            ".csv": [],
            ".pptx": [],
            ".txt": [],
            ".json": [],
        }

        path_list = []
        for path, subdirs, files in os.walk(self.path_data):
            for name in files:
                path_list.append(os.path.join(path, name))
                #print(os.path.join(path, name))

        # Loop through the path_list and categorize files
        for filename in path_list:
            file_extension = pathlib.Path(filename).suffix
            #print("File Extension:", file_extension)
            
            if file_extension in extension_lists:
                extension_lists[file_extension].append(filename)
        return extension_lists
    
    def upload_docx(self, extension_lists):
        #word
        data_docxs = []
        for doc in extension_lists[".docx"]:
            loader = Docx2txtLoader(doc)
            data = loader.load()
            data_docxs.extend(data)
        return data_docxs
    
    def upload_pdf(self, extension_lists):
        #pdf 
        data_pdf = []
        for doc in extension_lists[".pdf"]:
            loader = PyPDFLoader(doc)
            data = loader.load_and_split()
            data_pdf.extend(data)
        return data_pdf
    
    def upload_html(self, extension_lists):
        #html 
        data_html = []
        for doc in extension_lists[".html"]:
            loader = UnstructuredHTMLLoader(doc)
            data = loader.load()
            data_html.extend(data)
        return data_html
    
    def upload_png_ocr(self, extension_lists):
        #png ocr
        data_png = []
        for doc in extension_lists[".png"]:
            loader = UnstructuredImageLoader(doc)
            data = loader.load()
            data_png.extend(data)
        return data_png 

    def upload_excel(self, extension_lists, dataframe):
        #png ocr
        data_excel = []
        #excel text
        if dataframe == True :
            #Excel dataframe
            for doc in extension_lists[".xlsx"]:
                max_length = 0 
                max_column = "" #this section define for page_content_column
                df = pd.read_excel(doc)
                for column in df:
                    column_length = df[column].astype(str).str.len().mean() #page_content_Column will define by mean value in each columns
                    if column_length > max_length :
                        max_length = column_length
                        max_column = column
                loader = DataFrameLoader(df, page_content_column = max_column) #page_content_column = content that you want to keep
                data = loader.load()
                for row in data:
                    row.metadata["source"] = doc
                data_excel.extend(data)
        else :
            #Excel text
            for doc in extension_lists[".xlsx"]:
                loader = UnstructuredExcelLoader(doc)
                data = loader.load()
                data_excel.extend(data)
        return data_excel 
    
    def upload_csv(self, extension_lists, dataframe):
        data_csv = []
        if dataframe == True :
            #csv dataframe
            for doc in extension_lists[".csv"]:
                max_length = 0 
                max_column = "" #this section define for page_content_column
                df = pd.read_csv(doc)
                for column in df:
                    column_length = df[column].astype(str).str.len().mean() #page_content_Column will define by mean value in each columns
                    if column_length > max_length :
                        max_length = column_length
                        max_column = column
                loader = DataFrameLoader(df, page_content_column = max_column) #page_content_column = content that you want to keep
                data = loader.load()
                for row in data:
                    row.metadata["source"] = doc
                data_csv.extend(data)
        else:
            #csv text
            for doc in extension_lists[".csv"]:
                loader = CSVLoader(doc)
                data = loader.load()
                data_csv.extend(data)
        return data_csv
    
    def upload_pptx(self, extension_lists):
        #power point
        data_pptx = []
        for doc in extension_lists[".pptx"]:
            loader = UnstructuredPowerPointLoader(doc)
            data = loader.load()
            data_pptx.extend(data)
        return data_pptx
    
    def upload_txt(self, extension_lists):
        #txt 
        data_txt = []
        for doc in extension_lists[".txt"]:
            loader = TextLoader(doc)
            data = loader.load()
            data_txt.extend(data)
        return data_txt

    def upload_json(self, extension_lists):
        #json 
        data_json = []
        for doc in extension_lists[".json"]:
            loader = JSONLoader(
                file_path=doc,
                jq_schema='.[]',
                text_content=False
                )
            data = loader.load()
            data_json.extend(data)
        return data_json
    
    def count_files(self, extension_lists):
        file_extension_counts = {}
        # Count the quantity of each item
        for ext, file_list in extension_lists.items():
            file_extension_counts[ext] = len(file_list)
        return print(f"number of file:{file_extension_counts}")
        # Print the counts
        # for ext, count in file_extension_counts.items():
        #     return print(f"{ext}: {count} file")

    def create_document(self, dataframe=True):
        documents = []
        extension_lists = self.prepare_filetype()
        self.count_files(extension_lists)
        
        upload_functions = {
            ".docx": self.upload_docx,
            ".pdf": self.upload_pdf,
            ".html": self.upload_html,
            ".png": self.upload_png_ocr,
            ".xlsx": self.upload_excel,
            ".csv": self.upload_csv,
            ".pptx": self.upload_pptx,
            ".txt": self.upload_txt,
            ".json": self.upload_json,
        }

        for extension, upload_function in upload_functions.items():
            if len(extension_lists[extension]) > 0:
                if extension == ".xlsx" or extension == ".csv":
                    data = upload_function(extension_lists, dataframe)
                else:
                    data = upload_function(extension_lists)
                documents.extend(data)
    
        return documents

def split_docs(documents,chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    sp_docs = text_splitter.split_documents(documents)
    return sp_docs

@st.cache_resource 
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "thenlper/gte-base",
                                       model_kwargs = {'device': 'cpu'})
    return embeddings

def main():

    data = []
    DB_FAISS_UPLOAD_PATH = "vectorstores/db_faiss"
    DB_FAISS_OMNISCIEN_PATH = "/home/sira/sira_project/meta-Llama2/vectorstores_clean_doc_gte-base/db_faiss"
    with st.sidebar:
        option = st.selectbox(
                "Document option",
                ("omniscien", "upload"),
                )
        if option == "upload" :
            with st.form(key="upload") :
                directory = st.text_input("Your path")
                web = st.text_input("""Website URL """)
                submit_button_web = st.form_submit_button(label='Submit')
                if submit_button_web:
                    data_web = read_url(web)
                    data_dir = UploadDoc(directory).create_document()
                    data_web.extend(data_dir)
                    data.extend(data_web)


        #create vector from upload 
        if len(data) > 0 :
            sp_docs = split_docs(documents = data)
            st.write(f"This document have {len(sp_docs)} chunks")
            embeddings = load_embeddings()
            with st.spinner('Wait for create vector'):
                db = FAISS.from_documents(sp_docs, embeddings)
                db.save_local(DB_FAISS_UPLOAD_PATH)
                st.write(f"Your model is already store in {DB_FAISS_UPLOAD_PATH}")

        if option == "omniscien" :
            embeddings = load_embeddings()
            db = FAISS.load_local(DB_FAISS_OMNISCIEN_PATH, embeddings)
            st.write(f"Your model is already existing in {DB_FAISS_OMNISCIEN_PATH}")
        
            
if __name__ == '__main__':
        main()


