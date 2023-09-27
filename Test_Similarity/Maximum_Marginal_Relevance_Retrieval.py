import torch
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import time
import re

# Make sure the model path is correct for your system!
def load_llm(device):
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    if device == "cuda":
        llm = LlamaCpp(
            model_path="/home/sira/sira_project/meta-Llama2/llama-2-7b-chat.ggmlv3.q8_0.bin",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            callback_manager=callback_manager,
            verbose=True,
            n_ctx=2048
        )
        print("cuda")
        return llm
    else :
        llm = LlamaCpp(
            model_path="/home/sira/sira_project/meta-Llama2/llama-2-7b-chat.ggmlv3.q8_0.bin",
            n_ctx=2048,
            callback_manager=callback_manager,
            verbose=True,
        )
        print("CPU")
        return llm

custom_prompt_template = """ Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context : {context}
Question : {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context',
                                                                              'question'])
    return prompt

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
            url = 'www.' + url
        if '/index' in url:
            url = url.split('/index')[0] 
        convert_urls.append(url)
    return convert_urls

def regex_source(answer):
    pattern = r"'source': '(.*?)'"
    matchs = re.findall(pattern, str(answer))
    convert_urls = convert_to_website_format(matchs)
    res_urls = check_duplicate(source_list=convert_urls)
    return res_urls

def load_embeddings(device):
    if device == "cuda" :
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {'device': 'cuda'})
    else :
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs = {'device': 'cpu'})
    return embeddings

def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_FAISS_PATH = "vectorstores/db_faiss"
    embeddings = load_embeddings(device)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm(device)
    qa_prompt = set_custom_prompt(custom_prompt_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_type="mmr", search_kwargs = {'k':5, 'fetch_k' : 10}), 
        return_source_documents = True,
        chain_type_kwargs = {"prompt":qa_prompt}) 

    while True :
        query = input("ASK ABOUT THE DOCS:")
        if input != "break":	
            start = time.time()
            response = qa_chain({'query': query})
            print(response["result"])
            print(response)
            # urls = regex_source(response)
            # for count, url in enumerate(urls):
            #     print(str(count+1)+":", url)
            end = time.time()
            print("Respone time:",int(end-start),"sec")
        
        else :
            print("Exit")
            break

if __name__ == '__main__':
	main()