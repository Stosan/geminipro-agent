# GeminiPro-AI Hybrid Agent-RAG System Server Documentation 🚀

## Overview 📖

The GeminiPro-AI Hybrid Agent-RAG System is a high-performance and scalable AI system integrating agentic capabilities with a robust RAG pipeline.

It comprises of an Agentic System layered with an always-available RAG pipeline.

It employs a layered architecture to ensure modularity, maintainability, and flexibility. The system is built to leverage Server Sent Event for streaming real-time communication.

## AI/ML Stack

### Languages and Frameworks

- 🐍 |⚡ **Python:** Our primary language for developing models and conducting experiments. Python's versatility and rich ecosystem enable rapid prototyping and seamless integration. Utilized for code optimization, leveraging its speed and efficiency to enhance application performance, responsiveness, and scalability.

- 🔥 **PyTorch:** Our framework of choice for deep learning research. PyTorch's dynamic computation graph and intuitive API empower us to build and train powerful neural networks.

- ⚡ **FastAPI:** A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. FastAPI is used to build the backend of the GeminiPro-AI Agent Server, providing a robust and efficient way to handle HTTP requests and real-time communication.

- 🤗 **HuggingFace:** A popular open-source platform for natural language processing (NLP) and machine learning. Hugging Face provides a wide range of pre-trained models and tools that can be easily integrated into our AI/ML stack for tasks such as text generation, sentiment analysis, and more.

- 🌲 **Pinecone:** A vector database that enables efficient similarity search and retrieval. Pinecone is used to store and query vector embeddings, facilitating fast and scalable retrieval of relevant information in our RAG pipeline.

- 🤖 **GeminiPro-AI:** A state-of-the-art Large Language Model.


### Infrastructure and Scaling

- 🚀 **GitHub Actions:** Our CI/CD workflows are automated with GitHub Actions, enabling streamlined building, testing, and deployment processes.

- ☁️ **Google AI Studio:** A suite of AI and machine learning services provided by Google Cloud Platform.

## Getting Started 🛠️

Follow these steps to set up and run the GeminiPro-AI Agent Server on your local machine:

### Prerequisites 📋

- Python 3.12+

### Setting Up Pinecone

To set up Pinecone as vectorDB, follow these steps:

1. **Sign Up and Get an API Key**:
   - Go to the [Pinecone website](https://www.pinecone.io/) and sign up for an account.
   - Once you have signed up, navigate to the API keys section in your Pinecone dashboard and generate a new API key.

2. **Install Pinecone Client**:
   - Install the Pinecone client library using pip:
     ```bash
     pip install pinecone-client
     ```

3. **Configure Environment Variables**:
   - Add your Pinecone API key to the `.env` file:
     ```
     PINECONE_API_KEY=your_pinecone_api_key
     ```

### Installation 💽

1. Clone the repository:

   ```bash
   git clone https://github.com/Stosan/geminipro-agent.git
   cd geminipro-agent
   ```

2. Create a virtual environment and activate it:

   ```bash
   # For Unix-based systems
   python -m venv venv
   source venv/bin/activate

   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables. Copy the `.env.example` file in the root directory to `.env` and modify the values as needed.

### Running the Service

#### Start with TESTS
To run the tests, Follow these steps:

1. **Install pytest**:
   If you haven't already installed `pytest`, you can do so using pip:
   ```bash
   pip install pytest
   ```

2. **Run the tests**:
   You can run the tests by executing the following command in the root directory of the project:
   ```bash
   pytest
   ```

   This will automatically discover and run all the test files in the project.

3. **View detailed output**:
   To view more detailed output while running the tests, you can use the `-v` (verbose) option:
   ```bash
   pytest -v
   ```

4. **Generate a test coverage report**:
   To generate a test coverage report, you can use the `pytest-cov` plugin. First, install the plugin:
   ```bash
   pip install pytest-cov
   ```

1. Start the FastAPI server:

   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. Open a browser and navigate to `http://localhost:8000` to access the chatview frontend.
3. Open a browser and navigate to `http://localhost:8000/docs` to access the FastAPI Swagger documentation.

### What Next? 🚀

After successfully running the GeminiPro-AI Agent Server, you can start interacting with the various endpoints provided by the API. Here are some steps you can take next:

1. **Upload a PDF Document**:
   - Use the `/upload-doc` endpoint to upload a PDF file. This endpoint saves the file to a temporary directory and returns the file location.
   - Example using `curl`:
     ```bash
     curl -X POST "http://localhost:8000/api/v1/upload-doc" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path_to_your_file.pdf"
     ```

2. **Chat with the AI Agent**:
   - Use the `/chat-stream` endpoint to send chat requests and receive responses from the AI agent.
   - Example using `curl`:
     ```bash
     curl -X POST "http://localhost:8000/api/v1/chat-stream" -H "accept: application/json" -H "Content-Type: application/json" -d '{"sentence": "Hello, how are you?"}'
     ```

3. **Explore the API Documentation**:
   - Visit `http://localhost:8000/docs` to explore the interactive API documentation provided by Swagger UI. This allows you to test the endpoints directly from your browser.

4. **Health Check**:
   - Use the `/health` endpoint to check the health status of the application.
   - Example using `curl`:
     ```bash
     curl -X GET "http://localhost:8000/health"
     ```


### Improvements 🛠️✨

1. **RAG**
 - implement query rewriter for more performant retrieval
 - implement an OCR model for edge-case docs
 - implement a more advanced textsplitter like splitonpage

2. **Agent**
 - implement a finetuned LLM for more domain specific reasoning
 - implement an instruction fine-tuned LLM

3. **Security**
- implement HTTPSRedirectMiddleware from the fastAPI class