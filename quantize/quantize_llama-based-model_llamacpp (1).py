import os
import subprocess

def is_git_installed():
    try:
        subprocess.run("git --version", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    

def install_git():
    try:
        subprocess.run("apt-get update", shell=True, check=True)
        subprocess.run("apt-get install git -y", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception("Failed to install Git")


def download_llamacpp():
    if not is_git_installed():
        print("Git is not installed. Installing Git...")
        install_git()
    
    subprocess.run(f"git clone https://github.com/ggerganov/llama.cpp", shell=True, check=True)

def download_llama_model(model_id):
    if not is_git_installed():
        print("Git is not installed. Installing Git...")
        install_git()
    
    subprocess.run(f"git lfs install", shell=True, check=True)
    subprocess.run(f"git clone https://huggingface.co/{model_id}", shell=True, check=True)


def quantize_model_llmcpp(model, q_method, outdir):
    # path to llama.cpp local repo
    llamabase = "llama.cpp"
    
    if not os.path.isdir(llamabase):
        download_llamacpp()

    ggml_version = "gguf"
        
    if not os.path.isdir(model):
        raise Exception(f"Could not find model dir at {model}")
    if not os.path.isfile(f"{model}/config.json"):
        raise Exception(f"Could not find config.json in {model}")
    
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isdir(os.path.join(llamabase, "quantize")):
        print("+++++Building llama.cpp+++++")
        subprocess.run(f"cd {llamabase} && git pull && make clean && LLAMA_CUBLAS=1 make", shell=True, check=True)
        subprocess.run(f"pip install sentencepiece", shell=True, check=True)

    fp16 = f"{outdir}/{model.lower()}.{ggml_version}.fp16.bin"
    print(f"+++++Making unquantized GGUF at {fp16}+++++")
    if not os.path.isfile(fp16):
        subprocess.run(f"python3 {llamabase}/convert.py {model} --outtype f16 --outfile {fp16}", shell=True, check=True)
    else:
        print(f"Unquantized GGUF already exists at: {fp16}")

    print("+++++Making quantization+++++")

    qtype = f"{outdir}/{model.lower()}.{ggml_version}.{q_method}.bin"
    print(f"Making {q_method} : {qtype}")
    subprocess.run(f"{llamabase}/quantize {fp16} {qtype} {q_method}", shell=True, check=True)


    # Delete FP16 GGUF when done making quantizations
    os.remove(fp16)

if __name__ == "__main__":    
    model_id = input("Enter the llama-based model id: ")
    model_name = model_id.split('/')[-1]

    if not os.path.isdir(model_name):
        download_llama_model(model_id)

    outdir_name = input("Entrer output directory name: ")
    q_method = input("Choose one of quant methods[q4_0, q4_1, q5_0, q5_1, q8_0]: ")

    quantize_model_llmcpp(model_name, q_method, outdir_name)

