# Import required modules
import time
from typing import Annotated
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio, gc, uvicorn, re
from src.config.pinecone import configure_pinecone_index
from src.dependencies import get_current_username
from src.api_models.chat_model import ChatRequest
from src.agent.llm import LLM_Model
from src.agent.toolkit.base import MastivTools
from src.inference import StreamConversation
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Request, UploadFile, HTTPException, Depends, status
from src.config.settings import get_setting
from fastapi.middleware.cors import CORSMiddleware
from src.config import appconfig
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from src.ragpipeline.run import run_pipeline, save_pdf_to_temp
from src.utilities.Printer import printer
from src.utilities.helpers import static_response_generator,static_responses

# Get application settings
settings = get_setting()

# Description for API documentation
description = f"""
{settings.API_STR} helps you do awesome stuff. 🚀
"""

# Garbage collect to free up resources
gc.collect()

api_llm = LLM_Model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application lifespan.
    This function initializes and cleans up resources during the application's lifecycle.
    """
    print(running_mode)
    print()
    configure_pinecone_index()
    # MongoDB configuration
    # MongoDBContextConfig()
    print()
    MastivTools()
    print()
    printer(" ⚡️🚀 AI Server::Started", "sky_blue")
    print()
    printer(" ⚡️🏎  AI Server::Running", "sky_blue")
    yield
    printer(" ⚡️🚀 AI Server::SHUTDOWN", "red")


# Create FastAPI app instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=description,
    openapi_url=f"{settings.API_STR}/openapi.json",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan,
)

# Configure for development or production mode
if appconfig.ENV == "development":
    running_mode = "  👩‍💻 🛠️  Running in::development mode"
    health_path = "/dev/health"
else:
    app.add_middleware(HTTPSRedirectMiddleware)
    running_mode = "  🏭 ☁  Running in::production mode"
    health_path = "/health"


# Origins for CORS
origins = ["*"]

# Add middleware to allow CORS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount a static files directory
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get(f"{settings.API_STR}", status_code=status.HTTP_200_OK)  
# endpoint for root URL
def APIHome():
    """
    Returns a dictionary containing information about the application.
    """
    return {
        "ApplicationName": app.title,
        "ApplicationOwner": "GeminiPro AI Agent",
        "ApplicationVersion": "3.0.0",
        "ApplicationEngineer": "Sam Ayo",
        "ApplicationStatus": "running...",
    }

@app.get(health_path, status_code=status.HTTP_200_OK)  
def APIHealth():
    """
    Returns a dictionary containing information about the application.
    """
    return "healthy"


@app.post(f"{settings.API_STR}/upload-doc", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file. The file is saved to a temporary directory and its location is printed.

    Args:
        file (UploadFile): The PDF file to be uploaded.

    Returns:
        dict: A dictionary containing the file location.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed.",
        )

    try:
        return await run_pipeline(file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e}",
        )


async def generate_events(stc,dumped_data:dict,sentence:str):
    async for chunk in stc.generate_response(dumped_data.get("userData"), sentence):
        yield f"data: {chunk}\n\n"

@app.post(f"{settings.API_STR}/chat-stream")
async def generate_response(
    data: ChatRequest,
    username: str = Depends(get_current_username),
) -> StreamingResponse:
    """Endpoint for chat requests.
    It uses the StreamingConversationChain instance to generate responses,
    and then sends these responses as a streaming response.
    :param data: The request data.
    """
    try:
        dumped_data = data.model_dump()
        sentence = dumped_data.get("sentence").strip()
        # Basic attack protection: remove "[INST]" or "[/INST]" or "<|im_start|>"from the sentence
        sentence = re.sub(r"\[/?INST\]|<\|im_start\|>|<\|im_end\|>", "", sentence)


        if sentence.lower() in static_responses:
            # Use the generator for static responses
            async for response in static_response_generator(sentence):
                return StreamingResponse(response, media_type="text/event-stream")

        stc = StreamConversation(llm=api_llm)
        if data.userData is not None:
            dumped_data = data.model_dump()
            sentence = dumped_data.get("sentence").strip()
            return StreamingResponse(generate_events(stc,dumped_data, sentence), media_type="text/event-stream")
        else:
            return StreamingResponse(generate_events(stc,{}, sentence), media_type="text/event-stream")
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e}",
        )




# Main function to run the FastAPI server
async def main():
    config = uvicorn.Config(
        app,
        port=8000,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


# Run the FastAPI server if this script is executed
if __name__ == "__main__":
    asyncio.run(main())
