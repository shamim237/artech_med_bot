import unittest
from fastapi.testclient import TestClient
from app import app

class TestAnswerEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_valid_question(self):
        response = self.client.post("/answer", json={"question": "What is an overactive bladder?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())

    def test_empty_question(self):
        # Empty string fails Pydantic validation (constr(min_length=1))
        response = self.client.post("/answer", json={"question": ""})
        self.assertEqual(response.status_code, 422)  # Pydantic validation error
        self.assertIn("should have at least 1 character", response.json()["detail"][0]["msg"])

    def test_whitespace_question(self):
        # Whitespace-only string passes Pydantic but fails our custom validation
        response = self.client.post("/answer", json={"question": "   "})
        self.assertEqual(response.status_code, 500)  # Internal server error from our validation
        self.assertEqual(response.json()["detail"], "Error processing request")

    def test_missing_question_field(self):
        response = self.client.post("/answer", json={})
        self.assertEqual(response.status_code, 422)  # FastAPI validation error

    def test_invalid_json(self):
        response = self.client.post("/answer", data="invalid json")
        self.assertEqual(response.status_code, 422)  # FastAPI validation error

if __name__ == '__main__':
    unittest.main()