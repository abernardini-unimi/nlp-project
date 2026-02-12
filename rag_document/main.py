import asyncio

from db.create_database import create_single_service
from db.functions import get_all_services_from_db

async def main():
    print("Starting MCP-Document architecture...")
    
    # 1. Database Creation
    existing_services = get_all_services_from_db()
    if not existing_services:    
        print("🔧 Creating sample database...")
        for _ in range(20):
            service_name, success = create_single_service()
            
    # 2. Initialize the Manager
    from src.manager import get_manager
    manager = get_manager()
    success = await manager.initialize_async()
    if not success:
        print("Error during manager initialization.")
        return

    print("✅ 1/3 Manager successfully initialized")

    # 3. Initialize the API Server
    from api.api import run_api
    print("🔧 Starting API server in background...")
    api_task = asyncio.create_task(run_api())
    await asyncio.sleep(1)  
    print("✅ 2/3 API started in background")

    # 4. Initialize the MCP Server
    from mcp_server.server import run_mcp_server
    print('🔧 Initializing MCP server...')
    mcp_task = asyncio.create_task(run_mcp_server())
    await asyncio.sleep(1) 
    print('✅ 3/3 MCP Server started in background')

    print('🚀 Initialization successful, servers are listening...')
    await asyncio.gather(api_task, mcp_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 System shutdown requested (CTRL+C)")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()