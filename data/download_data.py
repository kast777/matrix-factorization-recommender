import os
import requests
import zipfile

def download_dataset():
    """
    Faz o download do dataset MovieLens 25M e extrai os ficheiros para a pasta data/.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    
    # Define caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    target_path = os.path.join(data_dir, "ml-25m.zip")
    extract_path = data_dir
    
    # Cria a pasta data se nao existir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Verifica se o dataset ja foi extraido
    if not os.path.exists(os.path.join(data_dir, "ml-25m")):
        print("Iniciando download do MovieLens 25M (aprox. 250MB)...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
            
            print("Download concluido. Extraindo ficheiros...")
            with zipfile.ZipFile(target_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Remove o arquivo zip para poupar espaco
            os.remove(target_path)
            print("Processo concluido com sucesso.")
            
        except Exception as e:
            print(f"Erro durante o processamento: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)
    else:
        print("Dataset ja encontrado em: data/ml-25m")

if __name__ == "__main__":
    download_dataset()
    