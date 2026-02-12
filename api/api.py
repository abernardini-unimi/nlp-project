import uvicorn # type: ignore
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore

from config.logger import logger
from config.settings import API_PORT, LOGGER_LEVEL

from api.routers import resources

app = FastAPI(
    title="MCP-Document API",
    description="API to manage database services and documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_api():
    """Start server FastAPI."""
    logger.info(f"🌐 Start server FastAPI on port {API_PORT}...")

    # Include i router prima di avviare il server
    app.include_router(resources.router)

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level=LOGGER_LEVEL.lower(),
        reload=False
    )
    server = uvicorn.Server(config)
    await server.serve() 
    return True


