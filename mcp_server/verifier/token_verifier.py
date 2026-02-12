from typing import Optional
from mcp.server.auth.provider import AccessToken, TokenVerifier  # type:ignore

from config.logger import logger
from src.manager import get_manager


class EnvironmentMultiServiceTokenVerifier(TokenVerifier):
    """Token verifier that dynamically maps unique tokens to services."""
    
    def __init__(self):
        """Initializes the token verifier."""
        logger.info("🔐 Token verifier initialized")

    @staticmethod
    def find_service_by_token(token: str) -> Optional[str]:
        """
        Searches for the service associated with the token in the current configuration.
        
        Args:
            token: The token to search for
            
        Returns:
            The service name and customer name if found, otherwise (None, None)
        """
        try:
            # Get the global manager instance
            manager = get_manager()

            for customer_name, service_token in manager.customer_tokens.items():
                for service_name, tokens in service_token.items():
                    if not isinstance(tokens, list):
                        logger.warning(f"⚠️ Tokens for {service_name} are not a valid list.")
                        continue
                    
                    if token in tokens:
                        logger.info(f"✅ Token found for service '{service_name}' of customer '{customer_name}'")
                        return service_name, customer_name
                
            logger.debug("❌ Token not found in any service")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Error while searching for token: {e}", exc_info=True)
            return None, None

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verifies the token and returns the associated service.
        Searches dynamically in the current configuration.
        """
        service_name, customer_name = self.find_service_by_token(token)
        if service_name is None or customer_name is None:
            logger.warning("❌ Invalid token or not associated with any service")
            return None
        
        client_id = f"{customer_name}-{service_name}"
        if service_name and customer_name:
            logger.info(f'✅ Token verified - CUSTOMER: {customer_name} - SERVICE: {service_name}')
            try:
                access_token = AccessToken(
                    token=token,
                    client_id=client_id,
                    scopes=["user"]
                )
                return access_token
            except Exception as e:
                logger.error(f"❌ Error creating AccessToken: {e}")
                return None
        
        logger.warning("❌ Invalid or unrecognized token")
        return None
