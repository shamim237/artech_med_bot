# Building a Retrieval-Augmented Question-Answering System with FastAPI and LangChain
Develop a simple question-answering web service that leverages Retrieval-Augmented Generation (RAG) to provide answers based on a set of provided documents. The service will be built using Python, FastAPI, and LangChain.

## Installation

1. Clone or download this repository
```
git clone https://github.com/shamim237/artech_med_bot.git
```
2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required packages:    
```
pip install -r requirements.txt
```

# Run the system

To run the system on your machine, follow the steps shown below:

## **preprocess.py**:
This script for preprocessing MSD (Excel) and CBIP (CSV) data files. This script provides standardized data cleaning and transformation operations for both file formats.

### Features
- Processes MSD Excel files and CBIP CSV files
- Standardizes text to lowercase
- Removes empty columns
- Handles missing values
- Eliminates duplicate rows
- Preserves original file structure
- Comprehensive logging

### Usage
```
python preprocess.py --msd-input "path/to/msd.xlsx" --cbip-input "path/to/cbip/directory"
```
#### Arguments
- `--msd-input`: Path to the MSD Excel file
- `--cbip-input`: Path to the directory containing CBIP CSV files


#### Output

The script creates a `processed_data` directory in your current working directory with the following structure:
processed_data/
├── msd/
│ └── msd_processed.csv
└── cbip/
└── [processed_csv_files]

### Data Processing Steps

1. **Text Standardization**: Converts all text to lowercase
2. **Column Cleaning**: Removes columns that are completely empty
3. **Missing Value Handling**: Fills NaN values with empty strings
4. **Duplicate Removal**: Removes duplicate rows from the dataset

### Error Handling

- The script includes comprehensive error handling and logging
- Errors are logged with timestamps and detailed messages
- Processing continues even if individual files fail


## "vectorize.py":
This script processes CSV documents and creates FAISS vector stores using LangChain and Hugging Face embeddings. It's designed to handle both MSD (Master Service Data) and medical data sources, converting them into efficient searchable vector representations.

### Features
- CSV document loading with support for multiple files
- Text chunking with configurable size and overlap
- FAISS vector store creation and persistence
- Comprehensive error handling and logging
- Support for Hugging Face embedding models

### Configuration

The script uses the following default configuration:

- MSD Data Path: `./processed_data/msd/msd_processed.csv`
- Medical CSV Path: `./processed_data/cbip/*.csv`
- MSD Vector Output: `./vectors_data/msd_data_vec`
- Medical Vector Output: `./vectors_data/med_data_vec`
- Embedding Model: `sentence-transformers/all-MiniLM-L12-v2`

### Usage
- Just run the script to get default output
```
python -m vectorize.py
```
- or change paths of the dataset

## "rag.py": 
This script implements a Retrieval-Augmented Generation (RAG) system using LangChain, FAISS vector store, and OpenAI's GPT-3.5 model. The system combines medical and general data sources to provide informed answers to user queries.

### Features

- Dual vector store integration (medical and general data)
- HuggingFace embeddings using `all-MiniLM-L12-v2` model
- OpenAI GPT-3.5 for answer generation
- Comprehensive error handling and logging
- Environment variable support for API keys

### Prerequisites

- OpenAI API key
    - Create a `.env` file in the project root and add your OpenAI API key: OPENAI_API_KEY=your_api_key_here
- Required vector stores in the `vectors_data` directory:
  - `msd_data_vec/` - General data vector store
  - `med_data_vec/` - Medical data vector store

### Usage
```
python rag.py
```

## "app.py":
This script is a FastAPI-based REST API that generates answers to questions using RAG (Retrieval-Augmented Generation) technology.

### Features

- Question answering endpoint with RAG integration
- Request ID tracking for all API calls
- Comprehensive error handling and logging
- Health check endpoint
- CORS support
- API documentation (Swagger UI and ReDoc)

### Usage
```
uvicorn app:app --reload
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Question Answering
- **Endpoint**: `/answer`
- **Method**: POST
- **Request Body**:
```
{
"question": "What is an overactive bladder?"
}
```
- **Response**:
```
{
"answer": "The generated answer..."
}
```

## test_rag.py:

The test suite validates the functionality of:
- Individual data retrievers (medicine and general data)
- Combined retriever functionality
- Answer generation system
- Error handling for edge cases

### Test Cases

The test suite includes the following test cases:

1. `test_data_retriever`: Tests retrieval from general data store
2. `test_med_retriever`: Tests retrieval from medical data store
3. `test_combined_retriever`: Tests the merged retriever functionality
4. `test_generate_answer`: Validates answer generation
5. `test_empty_query`: Tests error handling for invalid inputs

### Usage
```
python -m unittest test_rag.py
```

### Vector Store Setup

The system expects two FAISS indices in the `vectors/` directory:
- `msd_data_vec`: General knowledge vector store
- `med_data_vec`: Medical knowledge vector store

Both indices use the `sentence-transformers/all-MiniLM-L12-v2` embedding model.

#### Notes

- Ensure all vector stores are properly initialized before running tests
- The system uses the MiniLM-L12-v2 model for embeddings
- Empty or whitespace-only queries will raise ValueError exceptions


## test_app.py:

The test suite (`test_app.py`) validates the `/answer` endpoint's response to different types of requests, ensuring proper handling of both valid and invalid inputs.

### Test Cases

The test suite includes the following test cases:

1. **Valid Question Test**
   - Verifies that the endpoint correctly processes a valid question
   - Expects a 200 status code and an answer in the response

2. **Empty Question Test**
   - Validates handling of empty string inputs
   - Expects a 422 status code (Pydantic validation error)

3. **Whitespace Question Test**
   - Checks handling of whitespace-only inputs
   - Expects a 500 status code with an error message

4. **Missing Question Field Test**
   - Verifies behavior when the question field is omitted
   - Expects a 422 status code (FastAPI validation error)

5. **Invalid JSON Test**
   - Tests handling of malformed JSON requests
   - Expects a 422 status code (FastAPI validation error)

### Usage
```
python -m unittest test_app.py
```

## Assumptions and Trade-offs:
I generated and stored vector embeddings separately for disease/MSD data and medicine/CBIP data, believing that this separation would enhance the LLM's performance.

## Comments:
The quality of responses from this RAG-based LLM can be further strengthened through the following steps:
- Organizing the disease-related dataset more systematically.
- Structuring the medicine-related dataset more effectively.
- Enhancing disease-treatment and drug recommendations through better-organized mappings.
