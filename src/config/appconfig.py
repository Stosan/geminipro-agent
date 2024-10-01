# Load .env file using:
from dotenv import load_dotenv
load_dotenv()
import os

ENV = os.getenv("PYTHON_ENV")
APP_PORT = os.getenv("PORT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AUTH_USER = os.getenv("AUTH_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
MONGO_HOST = os.getenv("DB_HOST")
MONGO_PORT = os.getenv("DB_PORT")
MONGO_USER = os.getenv("DB_USER")
MONGO_PASSWORD = os.getenv("DB_PASSWORD")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "masteryhive-index"
