from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone


class PDFProcessor:
    @staticmethod
    async def process_pdf(file_path: str) -> List:
        raw_documents = PyPDFLoader(file_path).lazy_load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )
        return text_splitter.split_documents(raw_documents)

    @staticmethod
    async def generate_embeddings(documents, index_name: str):
        # modelPath = "sentence-transformers/all-MiniLM-l6-v2" # uses 384 dimension which is too small so i will be switching to mixedbread large
        modelPath = "mixedbread-ai/mxbai-embed-large-v1"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        vectordb = Pinecone.from_documents(documents, embeddings, index_name=index_name)
