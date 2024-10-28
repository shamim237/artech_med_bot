import unittest
from rag import load_faiss_index, generate_answer
from langchain.retrievers import MergerRetriever

class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize FAISS indices 
        cls.data_vec = load_faiss_index("./vectors_data/msd_data_vec", "sentence-transformers/all-MiniLM-L12-v2")
        cls.med_vec = load_faiss_index("./vectors_data/med_data_vec", "sentence-transformers/all-MiniLM-L12-v2")
        
        # Initialize retrievers 
        cls.data_retriever = cls.data_vec.as_retriever()
        cls.med_retriever = cls.med_vec.as_retriever()
        # Combine both retrievers into a single retriever
        cls.combined_retriever = MergerRetriever(retrievers=[cls.data_retriever, cls.med_retriever])

    def test_data_retriever(self):
        # Test the data retriever with a specific query
        query = "what are the symptoms of diabetes?"
        docs = self.data_retriever.get_relevant_documents(query)
        
        # Assert that documents are returned and are not empty
        self.assertIsNotNone(docs)
        self.assertTrue(len(docs) > 0)
        # Check if documents have content
        self.assertTrue(all(doc.page_content.strip() != "" for doc in docs))

    def test_med_retriever(self):
        # Test the medical retriever with a specific query
        query = "what are common antibiotics?"
        docs = self.med_retriever.get_relevant_documents(query)
        
        # Assert that documents are returned and are not empty
        self.assertIsNotNone(docs)
        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(doc.page_content.strip() != "" for doc in docs))

    def test_combined_retriever(self):
        # Test the combined retriever with a specific query
        query = "what is the treatment for high blood pressure?"
        docs = self.combined_retriever.get_relevant_documents(query)
        
        # Assert that documents are returned and are not empty
        self.assertIsNotNone(docs)
        self.assertTrue(len(docs) > 0)
        self.assertTrue(all(doc.page_content.strip() != "" for doc in docs))

    def test_generate_answer(self):
        # Test the answer generation function with a specific query
        query = "what are the side effects of aspirin?"
        response = generate_answer(query)
        
        # Assert that a valid response is returned
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

def test_empty_query(self):
    # Test the answer generation function with an empty query
    with self.assertRaises(ValueError):  # More specific exception
        generate_answer("")
    
    # Test the answer generation function with a whitespace-only query
    with self.assertRaises(ValueError):
        generate_answer("   ")  # Test whitespace-only query

if __name__ == '__main__':
    unittest.main()
