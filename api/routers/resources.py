from fastapi import APIRouter, status  # type: ignore
from config.logger import logger

from src.manager import get_manager
from api.schemas.classes import SaveServicePayload, DeleteServicePayload, RunTestPayload

router = APIRouter(
    prefix="/api/mcp",
    tags=["Resources"]
)

@router.post("/save_service", status_code=status.HTTP_201_CREATED)
async def save_service(service: SaveServicePayload):
    logger.info(f"CALL save_service API - data: {service}")
    try:
        manager = get_manager()
        success = await manager.save_service(service=service)

        if not success:
            return {
                "result": False,
                "message": "Error while saving the service"
            }
        return {
            "result": True,
            "message": "Service saved successfully"
        }
    except Exception as e:
        logger.error(f"Error in save_service: {e}", exc_info=True)
        return {
            "result": False,
            "message": "Error while saving the service"
        }


@router.post("/delete_service", status_code=status.HTTP_200_OK)
async def delete_service(payload: DeleteServicePayload):
    logger.info(f"CALL delete_service API - data: {payload.service_name}, {payload.customer_name}")
    try:
        manager = get_manager()
        success = await manager.delete_service(
            service_name=payload.service_name,
            customer_name=payload.customer_name
        )

        if not success:
            return {
                "result": False,
                "message": "Error while deleting the service"
            }
        return {
            "result": True,
            "message": "Service deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error in delete_service: {e}", exc_info=True)
        return {
            "result": False,
            "message": "Error while deleting the service"
        }
