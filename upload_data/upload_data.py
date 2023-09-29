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
from langchain.vectorstores import FAISS
 
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
from langchain.embeddings import HuggingFaceEmbeddings
# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# def save_uploadedfile(uploadedfile):
#     with open(os.path.join("Data", uploadedfile.name), "wb") as f:
#         f.write(filebytes)
#     return st.success("Saved File:{} to Data".format(uploadedfile.name))

def read_url(web):
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
    
    # def upload_png_ocr(self, extension_lists):
    #     #png ocr
    #     data_png = []
    #     for doc in extension_lists[".png"]:
    #         loader = UnstructuredImageLoader(doc)
    #         data = loader.load()
    #         data_png.extend(data)
    #     return data_png 

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
            #".png": self.upload_png_ocr,
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


def main():

    data = []
    print("""Please define your URL or path direcotory. Define "No" if you dont want import""")
    web = input('Your Website URL: ')
    directory = input('Your directory: ')
    saving_path = input("Which folder for save create vector file: ")
    if web != "" or  web != "no":
        data_web = read_url(web)
        data.extend(data_web)

    if directory != "" or  directory != "no":
        data_dir = UploadDoc(directory).create_document()
        data.extend(data_dir)

    if data is not None:
        sp_docs = split_docs(documents = data)
        print(f"Example document {sp_docs[:3]}")
        print(f"This document have {len(sp_docs)} chunks")

        print("Create Embedding...")
        embedding_function = HuggingFaceEmbeddings(model_name = "thenlper/gte-base",
                            model_kwargs = {'device': 'cpu'})
        db = FAISS.from_documents(sp_docs, embedding_function)
        print("Saving...")  
        db.save_local(saving_path) 
        print("Finish...")            

if __name__ == '__main__':
        main()