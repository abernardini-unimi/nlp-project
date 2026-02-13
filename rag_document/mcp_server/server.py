from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
import argparse
import asyncio
import sys
import time

from pydantic import AnyHttpUrl  # type: ignore
from starlette.requests import Request  # type: ignore
from mcp.server.auth.settings import AuthSettings  # type: ignore
from mcp.server.fastmcp import FastMCP  # type: ignore
from fastmcp.server.dependencies import get_http_request  # type: ignore

from config.logger import logger
from config.settings import MCP_PORT
from mcp_server.verifier.token_verifier import EnvironmentMultiServiceTokenVerifier
from src.manager import get_manager


# Get the singleton instance of the manager
manager = get_manager()

# --- FastMCP Server Configuration ---
mcp = FastMCP(
    "mcp-server",
    token_verifier=EnvironmentMultiServiceTokenVerifier(),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl("https://auth.example.com"),
        resource_server_url=AnyHttpUrl(f"http://localhost:{MCP_PORT}"),
        required_scopes=["user"],
    ),
    host="0.0.0.0",
    port=MCP_PORT,
)

# --- MCP Tools ---
@mcp.tool()
async def search_documents(query: str) -> Dict[str, Any]:
    """
    Search documents for a specific service using the RAG pipeline.

    Args:
        query: The search query

    Returns:
        A dictionary containing search results and metadata
    """
    logger.info(f"🔍 search_documents called with query: {query}")
    start_time = time.time()

    try:
        request: Request = get_http_request()
        auth_header = request.headers.get("authorization")
    except Exception as e:
        logger.error(f"❌ Error accessing HTTP request: {e}")
        auth_header = None

    # Extract Bearer token
    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.split(" ")[1]
        logger.info(f"🔑 Bearer token extracted: {bearer_token[:10]}...")
    else:
        logger.warning("❌ Authorization header missing or invalid")
        bearer_token = None

    # Map the token to the service using the current configuration
    service_name = None
    customer_name = None
    if bearer_token:
        try:
            service_name, customer_name = EnvironmentMultiServiceTokenVerifier.find_service_by_token(bearer_token)
        except Exception as e:
            logger.error(f"❌ Error while searching for service: {e}")

    if service_name is None or customer_name is None:
        error_msg = "❌ Invalid token or not associated with any service"
        logger.error(error_msg)
        return {
            "service_name": None,
            "query": query,
            "results": [],
            "total_found": 0,
            "execution_time": f"{time.time() - start_time:.2f}s",
            "status": "error",
            "error": error_msg,
        }

    # Perform search for the correct service
    try:
        results = await manager.search_query(
            service_name=service_name,
            customer_name=customer_name,
            query=query,
        )

        search_time = time.time() - start_time

        return {
            "service_name": service_name,
            "query": query,
            "results": results,
            "total_found": len(results),
            "execution_time": f"{search_time:.2f}s",
            "status": "success",
        }

    except Exception as e:
        error_msg = f"❌ Error during search for service {service_name}: {str(e)}"
        logger.error(error_msg)
        search_time = time.time() - start_time

        return {
            "service_name": service_name,
            "query": query,
            "results": [],
            "total_found": 0,
            "execution_time": f"{search_time:.2f}s",
            "status": "error",
            "error": error_msg,
        }


def _run_mcp_sync(transport: str):
    """Blocking function to run MCP (called from a separate thread)."""
    if transport in ("http", "streamable-http"):
        logger.info(f"🚀 Starting MCP server on port {MCP_PORT} with streamable-http transport")
        mcp.run(transport="streamable-http")
    else:
        logger.info(f"🚀 Starting MCP server on port {MCP_PORT} with SSE transport")
        mcp.run(transport="sse")


async def run_mcp_server(transport: str = "sse"):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    await loop.run_in_executor(executor, _run_mcp_sync, transport)
    return True



# --- Main ---
if __name__ == "__main__":
    try:
        run_mcp_server()
    except KeyboardInterrupt:
        logger.info("🛑 MCP Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in MCP server: {e}")
        sys.exit(1)
