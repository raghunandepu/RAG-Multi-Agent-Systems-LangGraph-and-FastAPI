# Model	              Dimensions	Cost per 1M tokens	Best For
# text-embedding-3-small	1536	    $0.02	            General use
# text-embedding-3-large	3072	    $0.13	                High accuracy
# text-embedding-ada-002	1536	    $0.10	                Legacy

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

#hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions#
#print(f"HuggingFace Embeddings dimension: {hf_embeddings.dimensions}")


# Ollama 
from langchain_ollama import OllamaEmbeddings
ollama_embeddings = OllamaEmbeddings(model="mistral") #Ollama Embeddings dimension: 4096
text = "This is a test sentence to check the dimensions of the Ollama embeddings."
embedding_vector = ollama_embeddings.embed_query(text)
print(f"Ollama Embeddings dimension: {len(embedding_vector)}")
