from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, Settings
from llama_index.embeddings.langchain.base import LangchainEmbedding
from llama_index.llms.ollama.base import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
import chromadb
import logging
import textwrap

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Step 1: Load PDF
print("Loading documents...")
documents = SimpleDirectoryReader(input_dir="pdfs", recursive=True).load_data()
print(f"Loaded {len(documents)} documents")

# Step 2: Define Embedder (using a multilingual embedding model for better cross-language performance)
print("Initializing embedding model...")
try:
    # Using BAAI's multilingual embedding model as mentioned in the paper
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )
except Exception:
    # Fallback to the original model if BAAI model isn't available
    print("Falling back to all-MiniLM-L6-v2 embeddings")
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

# Step 3: Initialize LLMs - main LLM and smaller LLM for relevance checking
print("Initializing LLMs...")
# Main LLM for final response generation
main_llm = Ollama(model="llama3:latest", request_timeout=120)

# Smaller LLM for relevance checking (use a smaller model for efficiency as mentioned in the paper)
# The paper used Llama 3.2 (3B) for relevance checking, but we'll use any available smaller model
relevance_llm = Ollama(model="llama3:latest", request_timeout=60)  # Ideally use a smaller model if available

# Step 4: Chroma vector store setup
print("Setting up vector store...")
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection("advanced-rag-legal")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 5: Build the index
print("Building index...")
Settings.llm = main_llm
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Step 6: Create retriever with MMR for diversity
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,  # Retrieve more candidates for filtering
    mmr_threshold=0.7    # Use Maximum Marginal Relevance for diversity
)

class AdvancedRAGQueryEngine:
    def __init__(self, retriever, relevance_llm, main_llm, embed_model):
        self.retriever = retriever
        self.relevance_llm = relevance_llm
        self.main_llm = main_llm
        self.embed_model = embed_model
        self.max_refinement_iterations = 3
        
    def relevance_check(self, query, retrieved_nodes):
        """Check relevance of retrieved nodes and filter out irrelevant ones."""
        if not retrieved_nodes:
            return [], query
        
        # Construct prompt for relevance check
        context_str = "\n\n".join([f"Document {i+1}:\n{node.text}" for i, node in enumerate(retrieved_nodes)])
        relevance_prompt = f"""
        [Task]: You are an intelligent legal assistant helping with relevance assessment.
        
        [User Query]: "{query}"
        
        [Retrieved Documents]:
        {context_str}
        
        [Instructions]:
        1. For each document, evaluate if it is relevant to answering the user query.
        2. Classify each document as "Relevant" or "Not Relevant".
        3. If you find that 50% or more of the documents are not relevant, suggest a refined query that might yield better results.
        
        [Output Format]:
        Relevance Assessment:
        - Document 1: [Relevant/Not Relevant]
        - Document 2: [Relevant/Not Relevant]
        ...
        
        Query Refinement: [Keep as is OR provide refined query]
        """
        
        # Get relevance assessment
        relevance_result = self.relevance_llm.complete(relevance_prompt)
        logger.info(f"Relevance Check Result:\n{relevance_result}")
        
        # Process the result to get relevant nodes and possibly refined query
        lines = relevance_result.text.strip().split('\n')
        relevant_indices = []
        refined_query = query
        
        in_assessment_section = False
        for line in lines:
            line = line.strip()
            
            # Check for relevance assessments
            if "Document" in line and ":" in line and ("Relevant" in line or "Not Relevant" in line):
                in_assessment_section = True
                try:
                    doc_idx = int(line.split("Document")[1].split(":")[0].strip()) - 1
                    if "Relevant" in line and "Not Relevant" not in line:
                        relevant_indices.append(doc_idx)
                except:
                    continue
            
            # Check for query refinement
            if "Query Refinement:" in line:
                in_assessment_section = False
                refined_part = line.split("Query Refinement:")[1].strip()
                if refined_part and "Keep as is" not in refined_part:
                    refined_query = refined_part
        
        # If we couldn't parse the format properly, default to using all nodes
        if not in_assessment_section and not relevant_indices:
            relevant_nodes = retrieved_nodes
        else:
            relevant_nodes = [retrieved_nodes[i] for i in relevant_indices if i < len(retrieved_nodes)]
        
        return relevant_nodes, refined_query
    
    def query(self, query_str):
        """Advanced RAG query with relevance checking and query refinement."""
        print(f"\nProcessing query: {query_str}")
        
        refined_query = query_str
        relevant_nodes = []
        iteration = 0
        
        # Iterative query refinement loop
        while iteration < self.max_refinement_iterations and not relevant_nodes:
            print(f"Refinement iteration {iteration+1}/{self.max_refinement_iterations}")
            print(f"Current query: {refined_query}")
            
            # Retrieve nodes
            retrieved_nodes = self.retriever.retrieve(refined_query)
            print(f"Retrieved {len(retrieved_nodes)} nodes")
            
            # Check relevance and refine query if needed
            relevant_nodes, new_refined_query = self.relevance_check(refined_query, retrieved_nodes)
            print(f"After relevance check: {len(relevant_nodes)} relevant nodes")
            
            # If no relevant nodes were found and we have a refined query
            if not relevant_nodes and new_refined_query != refined_query:
                refined_query = new_refined_query
                print(f"Query refined to: {refined_query}")
                iteration += 1
            else:
                break
        
        # If still no relevant nodes after all iterations, use the original retrieved nodes
        if not relevant_nodes and retrieved_nodes:
            print("No relevant nodes found after refinement. Using original retrieved nodes.")
            relevant_nodes = retrieved_nodes
        
        # Generate response using the final set of relevant nodes
        response_synthesizer = get_response_synthesizer(
            response_mode="refine",
            llm=self.main_llm
        )
        
        if not relevant_nodes:
            return "I couldn't find relevant information to answer your question."
        
        # Construct a prompt that includes the refined query for response generation
        response = response_synthesizer.synthesize(
            query=refined_query,
            nodes=relevant_nodes
        )
        
        # Display information about the retrieval process
        print(f"\nQuery processing complete:")
        print(f"Original query: {query_str}")
        print(f"Final query used: {refined_query}")
        print(f"Number of relevant chunks used: {len(relevant_nodes)}")
        
        return response

# Step 7: Create the advanced query engine
print("Creating advanced RAG query engine...")
query_engine = AdvancedRAGQueryEngine(
    retriever=retriever,
    relevance_llm=relevance_llm,
    main_llm=main_llm,
    embed_model=embed_model
)

# Step 8: Ask questions
print("\nAdvanced RAG System ready for questions!")
print("This system implements the approach from the paper 'LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval'")

while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == 'exit':
        break
    
    response = query_engine.query(user_query)
    
    # Format the response for better readability
    print("\nAnswer:")
    wrapped_text = textwrap.fill(str(response), width=100)
    print(wrapped_text)
