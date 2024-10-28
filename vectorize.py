import glob
import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorize.log'),
        logging.StreamHandler()
    ]
)

class VectorizationError(Exception):
    """Custom exception for vectorization-related errors"""
    pass

def load_csv_documents(csv_file_path: str) -> List[Document]:
    """
    Load CSV documents from the specified path.

    Args:
        csv_file_path (str): Path pattern to search for CSV files.

    Returns:
        List[Document]: A list of documents loaded from the CSV files.

    Raises:
        VectorizationError: If no CSV files are found or if there's an error loading them.
    """
    try:
        documents = []
        csv_files = list(glob.glob(csv_file_path))
        
        if not csv_files:
            raise VectorizationError(f"No CSV files found at path: {csv_file_path}")
        
        for csv_file in csv_files:
            logging.info(f"Loading CSV file: {csv_file}")
            loader = CSVLoader(csv_file, encoding="utf-8")
            documents.extend(loader.load())
            
        logging.info(f"Successfully loaded {len(documents)} documents from {len(csv_files)} CSV files")
        return documents
    
    except Exception as e:
        raise VectorizationError(f"Error loading CSV documents: {str(e)}")

def create_vector_store(
    documents: List[Document],
    embeddings_model: HuggingFaceEmbeddings,
    output_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Optional[FAISS]:
    """
    Create and save a FAISS vector store from documents.

    Args:
        documents (List[Document]): List of documents to vectorize
        embeddings_model (HuggingFaceEmbeddings): The embeddings model to use
        output_path (str): Path to save the FAISS index
        chunk_size (int, optional): Size of text chunks. Defaults to 500.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 50.

    Returns:
        Optional[FAISS]: The created FAISS index if successful, None otherwise
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        
        faiss_index = FAISS.from_documents(chunked_documents, embeddings_model)
        
       
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss_index.save_local(output_path)
        logging.info(f"Successfully saved FAISS index to {output_path}")
        
        return faiss_index
    
    except Exception as e:
        logging.error(f"Error creating vector store: {str(e)}")
        return None

def main():
    try:
        # Configuration with relative paths
        config = {
            'msd_data_path': "./processed_data/msd/msd_processed.csv",
            'medical_csv_path': "./processed_data/cbip/*.csv",
            'msd_vector_path': "./vectors_data/msd_data_vec",
            'medical_vector_path': "./vectors_data/med_data_vec",
            'model_name': "sentence-transformers/all-MiniLM-L12-v2"
        }
        
        # Create vectors_data directory if it doesn't exist
        Path("./vectors_data").mkdir(exist_ok=True)
        
        logging.info("Starting vectorization process")
        
        # Load documents
        msd_data_documents = load_csv_documents(config['msd_data_path'])
        medical_documents = load_csv_documents(config['medical_csv_path'])
        
        # Initialize embeddings model
        logging.info(f"Initializing embeddings model: {config['model_name']}")
        embeddings_model = HuggingFaceEmbeddings(model_name=config['model_name'])
        
        # Create vector stores
        msd_index = create_vector_store(
            msd_data_documents,
            embeddings_model,
            config['msd_vector_path']
        )
        
        medical_index = create_vector_store(
            medical_documents,
            embeddings_model,
            config['medical_vector_path']
        )
        
        if msd_index and medical_index:
            logging.info("Vectorization process completed successfully")
        else:
            logging.error("Vectorization process completed with errors")
            
    except VectorizationError as ve:
        logging.error(f"Vectorization error: {str(ve)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()