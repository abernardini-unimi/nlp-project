from db.connection import Neo4jDatabaseConnection

from config.logger import logger
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from schemas.classes import Service


try:
    db = Neo4jDatabaseConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    logger.info("✅ Successfully connected to the Neo4j database")
except Exception as e:
    logger.error(f"❌ Error connecting to the Neo4j database: {e}")
    db = None


def get_service_by_name_from_db(service_name: str, customer_name: str):
    """Retrieve a service from the database by name and customer"""
    if db is None:
        logger.error("❌ Database connection not available")
        return None

    service_name = service_name.strip().lower()
    customer_name = customer_name.strip().lower()

    try:
        query = """
        MATCH (s:Service {name: $service_name})-[:HAS_CUSTOMER]->(c:Customer {name: $customer_name})
        OPTIONAL MATCH (s)-[:HAS_DOCUMENT]->(d:Document)
        RETURN 
            s.name AS name, 
            s.description AS description,
            s.tokens AS tokens,
            c.name AS customer_name,
            collect(d) AS documents
        """
        result = db.run_query(query, {
            'customer_name': customer_name,
            'service_name': service_name
        })

        if not result:
            logger.warning(f"⚠️ Service '{service_name}' for customer '{customer_name}' not found in the database")
            return None
        
        return result[0]
    except Exception as e:  
        logger.error(f"❌ Error retrieving services: {e}")
        return None


def get_all_services_from_db():
    """Retrieve all services from the database"""
    if db is None:
        logger.error("❌ Database connection not available")
        return []
    
    try:
        query = """
        MATCH (s:Service)-[:HAS_CUSTOMER]->(c:Customer)
        OPTIONAL MATCH (s)-[:HAS_DOCUMENT]->(d:Document)
        RETURN 
            s.name AS name, 
            s.description AS description,
            s.tokens AS tokens,
            c.name AS customer_name,
            collect(DISTINCT d) AS documents
        """
        results = db.run_query(query)
        return results
    except Exception as e:  
        logger.error(f"❌ Error retrieving services: {e}")
        return []


def create_services_in_db(service: Service) -> bool:
    """
    Creates a service for a customer. 
    If the service already exists for that customer, it is first deleted (using delete_service) and then recreated from scratch.
    """
    if db is None:
        logger.error("❌ Database connection not available")
        return False

    try:
        service_name = service.name.strip().lower()
        customer_name = service.customer_name.strip().lower()

        # Check if it already exists
        existing = get_service_by_name_from_db(service_name, customer_name)
        if existing:
            success = delete_service_from_db(service_name, customer_name)
            if not success:
                logger.error("Error deleting existing service")
                return False

        # Create the new service from scratch
        create_query = """
        MERGE (c:Customer {name: $customer_name})
        CREATE (s:Service {
            name: $service_name,
            description: $service_description,
            tokens: $service_tokens
        })-[:HAS_CUSTOMER]->(c)
        WITH s
        UNWIND $documents AS doc
        CREATE (d:Document {
            name: doc.name,
            chunk_size: doc.chunk_size,
            chunk_overlap: doc.chunk_overlap,
            min_chunk_size: doc.min_chunk_size
        })
        CREATE (s)-[:HAS_DOCUMENT]->(d)
        """

        doc_list = [{
            'name': doc.name.strip().lower(),
            'chunk_size': doc.chunk_size,
            'chunk_overlap': doc.chunk_overlap,
            'min_chunk_size': doc.min_chunk_size
        } for doc in service.documents]

        db.run_query(create_query, {
            'customer_name': customer_name,
            'service_name': service_name,
            'service_description': service.description.strip(),
            'service_tokens': service.tokens,
            'documents': doc_list
        })

        logger.info(f"✅ Service '{service.name}' for customer '{customer_name}' created/updated successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Error creating service {service.name}: {e}")
        return False


def delete_service_from_db(service_name: str, customer_name: str) -> bool:
    """
    Deletes a specific service for a customer from the database.
    """
    if db is None:
        logger.error("❌ Database connection not available")
        return False
    
    service_name = service_name.strip().lower()
    customer_name = customer_name.strip().lower()
    
    try:
        result = get_service_by_name_from_db(service_name, customer_name)
        if not result:
            logger.warning(f"⚠️ Service '{service_name}' for customer '{customer_name}' not found")
            return True  # Nothing to delete
        
        # Delete the service
        query = f"""
            MATCH (s:Service {{name : '{service_name}'}})-[:HAS_CUSTOMER]->(C:Customer {{name : '{customer_name}'}})
            DETACH DELETE s
        """
        db.run_query(query)

        # Delete any orphan nodes
        success = delete_unconnected_nodes()
        if not success:
            logger.error("❌ Error deleting orphan nodes")
            return False
        
        logger.info(f"✅ Successfully deleted service '{service_name}' for customer '{customer_name}'")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error deleting service '{service_name}': {e}")
        return False


def delete_unconnected_nodes():
    try:
        if db is None:
            logger.error("❌ Database connection not available")
            return False
    
        query = """
            MATCH (n)
            WHERE NOT (n)--()
            DETACH DELETE n
            RETURN count(n) AS deleted_count
        """
        result = db.run_query(query)
        deleted_count = result[0].get('deleted_count', 0) if result else 0
        
        logger.info(f"✅ Deleted {deleted_count} unconnected nodes.")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error deleting unconnected nodes: {e}")
        return False
    
    
def delete_all_services_from_db() -> bool:
    """
    Deletes all services and customers from the database.
    """
    if db is None:
        logger.error("❌ Database connection not available")
        return False
    
    try:
        query = "MATCH (n) DETACH DELETE n"
        db.run_query(query)        
        logger.info("✅ All services and customers successfully deleted from the database")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error deleting all services: {e}")
        return False
