from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, Settings
from llama_index.embeddings.langchain.base import LangchainEmbedding
from llama_index.llms.ollama.base import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
import chromadb

# Step 1: Load PDF
documents = SimpleDirectoryReader(input_dir="pdfs", recursive=True).load_data()

# Step 2: Define Embedder
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Step 3: Initialize LLaMA 3 via Ollama
llm = Ollama(model="llama3:latest", request_timeout=120)

# Step 4: Chroma vector store setup
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection("rag-pdf")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 5: Build the index
Settings.llm = llm
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Step 6: Create a query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# Step 7: Ask questions
while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == 'exit':
        break
    response = query_engine.query(user_query)
    print("\nAnswer:\n", response)