# import library
from langchain_google_genai import ChatGoogleGenerativeAI



def LLM_Model():
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9)
    
    # Return the initialized model
    return llm
