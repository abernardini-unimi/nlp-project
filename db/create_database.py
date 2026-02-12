import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from schemas.classes import Service, Document
from db.functions import create_services_in_db

customers = {
    "increso": ["increso.pdf"],
    "hotel": ["hotel.pdf"],
    "assicurazione": ["unipol.pdf"],
    "fornitore_gas_luce": ["utenza.txt", "pagamento.txt", "fatturazione.txt", "rateizzazione.txt"]
}

def generate_random_service():
    """Generates example services with random data and prepares them for database insertion"""
    customer_key = random.choice(list(customers.keys()))
    documents = customers[customer_key]
    number = random.randint(1, 10000)

    num_tokens = random.randint(1, 2)

    # generate unique tokens
    tokens = {f"token_{random.randint(1, 999999):06d}" for _ in range(num_tokens)}

    # ensure complete uniqueness in rare cases
    while len(tokens) < num_tokens:
        tokens.add(f"token_{random.randint(1, 999999):06d}")

    service = Service(
        name=f"{customer_key}_{number}",
        description=f"Service description {customer_key}_{number}",
        tokens=list(tokens),
        customer_name=customer_key,
        documents=[]
    )

    for doc in documents:
        chunk_size = random.randint(200, 1500)
        chunk_overlap = random.randint(50, chunk_size // 2)
        min_chunk_size = random.randint(chunk_overlap, chunk_size)

        document = Document(
            name=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )

        service.documents.append(document)

    return service


def create_single_service():
    """Generates and creates a single service in the database."""
    service = generate_random_service()
    success = create_services_in_db(service)

    return service.name, success


def generate_random_services(num_services: int) -> list[Service]:
    """Generates a list of random services."""
    services = []
    for _ in range(num_services):
        service = generate_random_service()
        services.append(service)
    return services
