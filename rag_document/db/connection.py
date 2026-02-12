from neo4j import GraphDatabase  # type: ignore
from neo4j.exceptions import ServiceUnavailable, AuthError  # type: ignore

from config.logger import logger

class Neo4jDatabaseConnection:
    """Class to manage the connection to the Neo4j database"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initializes the connection to the Neo4j database
        
        Args:
            uri: Database URI (e.g., "bolt://localhost:7687")
            user: Username
            password: Password
        """
        try:
            logger.info(f"Attempting to connect to Neo4j database: {uri}")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify the connection
            self.driver.verify_connectivity()
            logger.info("Successfully connected to the Neo4j database")
        except AuthError as e:
            logger.error(f"Authentication error: {e}")
            exit(1)
        except ServiceUnavailable as e:
            logger.error(f"Database unavailable: {e}")
            exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            exit(1)
    
    def run_query(self, query: str, parameters: dict = None):
        """
        Executes a Cypher query on the database
        
        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query
            
        Returns:
            The results of the query
        """
        try:
            with self.get_session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def close(self):
        """Closes the database connection"""
        try:
            if hasattr(self, 'driver'):
                self.driver.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing the database connection: {e}")
    
    def get_session(self):
        """Returns a database session"""
        try:
            return self.driver.session()
        except Exception as e:
            logger.error(f"Error creating database session: {e}")
            raise
