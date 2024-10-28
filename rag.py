import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import MergerRetriever

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_faiss_index(folder_path: str, model_name: str) -> Optional[FAISS]:
    """
    Load a FAISS index with a specific embedding model.
    
    Args:
        folder_path: Path to the FAISS index folder
        model_name: Name of the HuggingFace embedding model
    
    Returns:
        FAISS: Loaded FAISS index object
        
    Raises:
        ValueError: If the folder path doesn't exist
    """
    try:
        if not os.path.exists(folder_path):
            raise ValueError(f"FAISS index folder not found: {folder_path}")
            
        logger.info(f"Loading FAISS index from {folder_path}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        raise

def generate_answer(query: str) -> str:
    """
    Generate an answer for the given query using RAG.
    
    Args:
        query: User's question
        
    Returns:
        str: Generated answer
        
    Raises:
        ValueError: If query is empty or required files are missing
    """
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Get the current directory and construct paths
        current_dir = Path(__file__).parent
        vectors_dir = current_dir / "vectors_data"
        
        # Validate vectors directory exists
        if not vectors_dir.exists():
            raise ValueError(f"Vectors directory not found at {vectors_dir}")
            
        # Load FAISS indices
        logger.info("Loading FAISS indices...")
        data_vec = load_faiss_index(
            str(vectors_dir / "msd_data_vec"),
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        med_vec = load_faiss_index(
            str(vectors_dir / "med_data_vec"),
            "sentence-transformers/all-MiniLM-L12-v2"
        )

        # Create the LLM instance
        logger.info("Initializing LLM...")
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define the prompt template
        template = """You are a helpful medical information assistant. Use the following pieces of context to answer the medical question at the end.
        
        Important notes:
        - Base your answer strictly on the provided context and understandable for all readers
        - If you don't know the answer, just say that you don't know
        - Include relevant disclaimers about consulting healthcare professionals
        - If suggesting medications (upon {question}), mention potential side effects if provided in the context
        - Highlight if the information is general knowledge or requires professional medical advice
        
        Context: {context}
        
        Question: {question}
        
        Medical Information Assistant:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], 
            template=template
        )

        # Initialize and combine retrievers
        logger.info("Setting up retrieval chain...")
        data_retriever = data_vec.as_retriever()
        med_retriever = med_vec.as_retriever()
        combined_retriever = MergerRetriever(
            retrievers=[data_retriever, med_retriever]
        )

        # Initialize the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=combined_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Run the chain
        logger.info("Generating answer...")
        result = qa_chain.invoke({"query": query})
        logger.info("Answer generated successfully")
        
        return result["result"]
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

def main():
    """
    Main function to demonstrate the usage of the RAG system.
    """
    try:
        # Example usage
        query = "suggest me some medicine for bronchitis"
        logger.info(f"Processing query: {query}")
        
        response = generate_answer(query)
        print("\nQuery:", query)
        print("\nResponse:", response)
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()