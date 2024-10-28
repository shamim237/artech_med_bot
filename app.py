import logging
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr, Field, validator, constr
from rag import generate_answer  # Importing the function to generate answers

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)  # Ensure logs directory is available for logging

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Question Answering API",  # Title of the API
    description="API for generating answers using RAG",  # Description of the API
    version="1.0.0",  # Version of the API
    docs_url="/docs",  # URL for API documentation
    redoc_url="/redoc"  # URL for ReDoc documentation
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; modify in production for security
    allow_credentials=True,  # Allow credentials to be included in requests
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[  # Handlers for logging output
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')  # Log to file
    ]
)
logger = logging.getLogger(__name__)  # Create a logger for this module

def generate_request_id() -> str:
    """Generate a unique request ID."""  # Docstring for function
    return str(uuid.uuid4())  # Return a new UUID as a string

class QuestionRequest(BaseModel):
    """Request model for question answering endpoint."""  # Docstring for model
    question: constr(min_length=1) = Field(  # Field for the question with validation
        ...,
        description="The question to be answered",  # Description of the field
        example="What is the capital of France?"  # Example question
    )
    
    @validator('question')  # Validator for the question field
    def validate_question(cls, v):
        if v.strip() == "":  # Check if the question is empty
            raise HTTPException(
                status_code=500,  # Raise HTTP 500 if empty
                detail="Error processing request"  # Error message
            )
        return v.strip()  # Return the trimmed question

class ErrorResponse(BaseModel):
    """Standard error response model."""  # Docstring for error response model
    detail: str  # Detail of the error
    request_id: Optional[str] = None  # Optional request ID for tracking

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Middleware to add request ID to all requests."""  # Docstring for middleware
    request_id = generate_request_id()  # Generate a new request ID
    request.state.request_id = request_id  # Store request ID in request state
    response = await call_next(request)  # Process the request
    response.headers["X-Request-ID"] = request_id  # Add request ID to response headers
    return response  # Return the response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler to include request ID."""  # Docstring for exception handler
    return JSONResponse(
        status_code=exc.status_code,  # Return the status code from the exception
        content={"detail": exc.detail, "request_id": request.state.request_id}  # Include error details and request ID
    )

@app.post("/answer",
    response_model=Dict[str, str],  # Response model for the endpoint
    summary="Generate an answer for the given question",  # Summary of the endpoint
    response_description="Returns the generated answer",  # Description of the response
    responses={  # Possible responses
        400: {"model": ErrorResponse},  # Bad request response
        500: {"model": ErrorResponse},  # Internal server error response
        504: {"model": ErrorResponse}   # Gateway timeout response
    }
)
async def get_answer(request: Request, question_request: QuestionRequest) -> Dict[str, str]:
    """
    Generate an answer for the given question.
    
    Args:
        request: FastAPI request object  # Description of request argument
        question_request: The question request model  # Description of question_request argument
        
    Returns:
        Dict containing the generated answer  # Description of return value
        
    Raises:
        HTTPException: For various error conditions  # Description of possible exceptions
    """
    request_id = request.state.request_id  # Retrieve request ID from state
    logger.info("Processing request %s - Question: %s", request_id, question_request.question)  # Log the processing request
    
    try:
        # Additional validation
        if len(question_request.question) > 1000:  # Check if question exceeds max length
            logger.warning("Request %s: Question exceeds maximum length", request_id)  # Log warning
            raise HTTPException(
                status_code=400,  # Raise HTTP 400 for bad request
                detail="Question length exceeds maximum allowed characters (1000)"  # Error message
            )

        # Convert generate_answer to async if it's not already
        async def async_generate_answer(question: str):
            loop = asyncio.get_event_loop()  # Get the current event loop
            return await loop.run_in_executor(None, generate_answer, question)  # Run the generate_answer function in executor

        # Generate answer with timeout handling
        answer = await asyncio.wait_for(
            async_generate_answer(question_request.question),  # Await the answer generation
            timeout=120.0  # 120 second timeout
        )
        
        logger.info("Request %s: Successfully generated answer", request_id)  # Log successful answer generation
        return {"answer": answer}  # Return the generated answer

    except asyncio.TimeoutError:  # Handle timeout errors
        logger.error("Request %s: Generation timeout", request_id)  # Log timeout error
        raise HTTPException(
            status_code=504,  # Raise HTTP 504 for timeout
            detail="Answer generation timed out"  # Error message
        )
        
    except ValueError as ve:  # Handle value errors
        logger.error("Request %s: Validation error - %s", request_id, str(ve))  # Log validation error
        raise HTTPException(
            status_code=400,  # Raise HTTP 400 for bad request
            detail=str(ve)  # Return the validation error message
        )
    
    except Exception as e:  # Handle all other exceptions
        logger.error(
            "Request %s: Unexpected error - %s",
            request_id,
            str(e),
            exc_info=True  # Include stack trace in logs
        )
        raise HTTPException(
            status_code=500,  # Raise HTTP 500 for internal server error
            detail="Internal server error occurred. Please try again later."  # Error message
        )

@app.get("/health",
    summary="Health check endpoint",  # Summary of the health check endpoint
    response_description="Returns the API health status"  # Description of the response
)
async def health_check():
    """Health check endpoint to verify API is running."""  # Docstring for health check
    return {"status": "healthy"}  # Return health status

@app.on_event("startup")
async def startup_event():
    """Initialize any resources on startup."""  # Docstring for startup event
    logger.info("Starting up Question Answering API")  # Log startup event
    # Add any initialization code here (e.g., loading models)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup any resources on shutdown."""  # Docstring for shutdown event
    logger.info("Shutting down Question Answering API")  # Log shutdown event
    # Add cleanup code here

if __name__ == "__main__":  # Entry point for the application
    import uvicorn  # Import Uvicorn for running the app
    
    # Load configuration from environment variables or use defaults
    host = "0.0.0.0"  # Host address
    port = 8000  # Port number
    
    logger.info(f"Starting server on {host}:{port}")  # Log server start
    
    uvicorn.run(
        "app:app",  # Application to run
        host=host,  # Host address
        port=port,  # Port number
        reload=False,  # Disable auto-reload in production
        log_level="info",  # Set log level
        access_log=True,  # Enable access logging
        workers=1  # Number of worker processes
    )
