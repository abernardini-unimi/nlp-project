from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import time
import os
import random
import pandas as pd
from datetime import datetime

from src.retrievers.i_retriever import IRetriever

from config.settings import BASE_VECTORSTORE_PATH, VECTORSTORE_PATH
from schemas.classes import Service, Document 
from src.pipeline import RAGPipeline

customers = {
    "increso": ["increso.pdf"],
    "hotel": ["hotel.pdf"],
    "assicurazione": ["unipol.pdf"],
    "fornitore_gas_luce": ["utenza.txt", "pagamento.txt", "fatturazione.txt", "rateizzazione.txt"],
    "whatsapp": ["whatsapp.pdf"],
    "etsy": ["Etsy.pdf"],
    "apple": ["apple.pdf"],
    "amazon": ["Amazon.docx"]
}

PAGES_MAPPING = {
    "Amazon.docx": 20, "apple.pdf": 193, "Etsy.pdf": 43, "rateizzazione.txt": 1,
    "hotel.pdf": 4, "increso.pdf": 19, "pagamento.txt": 1, "fatturazione.txt": 1,
    "unipol.pdf": 52, "utenza.txt": 1, "whatsapp.pdf": 63
}

def _get_service_vectorstore_path(service_name: str, customer_name: str) -> Path:
    return Path(VECTORSTORE_PATH) / customer_name.strip() / service_name.strip()


async def create_single_pipeline(retriever: IRetriever, service: Service) -> Tuple[Optional[RAGPipeline], float]:
    """
    Creates a RAG pipeline and returns the pipeline object and the time elapsed.
    Returns: (pipeline, elapsed_time_seconds)
    """
    start_time = time.time()
    
    try:
        pipeline = RAGPipeline(retriever.service_name, retriever.customer_name, retriever=retriever)

        # Loading documents
        upload_docs = await pipeline.load_documents_batch(service.documents)
        if len(upload_docs) == 0:
            print(f"    ❌ No documents loaded for {service.name}")
            return None, 0.0
        
        # Create index
        success = await pipeline.build_index_async()
        if not success:
            print(f"    ❌ Error creating index for {service.name}")
            return None, 0.0

        # Save to disk
        service_path = _get_service_vectorstore_path(service.name, service.customer_name)
        service_path.mkdir(parents=True, exist_ok=True)
        success = await pipeline.save_pipeline(str(service_path))
        if not success:
            print(f"    ❌ Error saving pipeline for {service.name}")
            return None, 0.0

        elapsed_time = time.time() - start_time
        print(f"    ✅ Pipeline created for {service.name} - Retriever: {type(retriever).__name__} in {elapsed_time:.2f}s")
        
        return pipeline, elapsed_time

    except Exception as e:
        print(f"    ❌ Exception during pipeline creation for {service.name}: {e}")
        return None, 0.0
    

def save_results_to_excel(results: Dict, output_path: str) -> Path:
    """Saves results to an Excel file with multiple sheets"""
    
    try:
        rows = []
        detailed_rows = []
        
        for key, value in results.items():
            service_name, customer_name, pipeline_name, question = key
            
            # Sheet 1: Summary (one row per query)
            top_score = value['results'][0][1] if value['results'] else 0
            num_results = len(value['results'])
            
            rows.append({
                'Service': service_name,
                'Customer': customer_name,
                'Retriever': pipeline_name,
                'Question': question,
                'Expected Answer': value.get('expected_answer', 'N/A'),
                'Top Score': round(top_score, 4),
                'Num Results': num_results,
                'Search Time (s)': round(value['search_time'], 4),
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Sheet 2: Detailed (one row for each result)
            for idx, (chunk, score) in enumerate(value['results'], 1):
                detailed_rows.append({
                    'Service': service_name,
                    'Customer': customer_name,
                    'Retriever': pipeline_name,
                    'Question': question,
                    'Rank': idx,
                    'Score': round(score, 4),
                    'Doc ID': chunk.doc_id,
                    'Content Preview': chunk.content[:300],
                    'Content Length': len(chunk.content),
                    'Search Time (s)': round(value['search_time'], 4)
                })
        
        # Create DataFrames
        df_summary = pd.DataFrame(rows)
        df_detailed = pd.DataFrame(detailed_rows)
        
        # Calculate statistics per retriever
        stats_rows = []
        if not df_summary.empty:
            for retriever in df_summary['Retriever'].unique():
                retriever_data = df_summary[df_summary['Retriever'] == retriever]
                
                stats_rows.append({
                    'Retriever': retriever,
                    'Total Queries': len(retriever_data),
                    'Avg Top Score': round(retriever_data['Top Score'].mean(), 4),
                    'Avg Search Time (s)': round(retriever_data['Search Time (s)'].mean(), 4),
                    'Min Search Time (s)': round(retriever_data['Search Time (s)'].min(), 4),
                    'Max Search Time (s)': round(retriever_data['Search Time (s)'].max(), 4),
                    'Total Results': retriever_data['Num Results'].sum()
                })
        
        df_stats = pd.DataFrame(stats_rows)
        
        # Save to Excel with multiple sheets
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed Results
            df_detailed.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Sheet 3: Statistics
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Auto-format column width
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\n💾 Results saved to Excel: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ Error saving results to Excel: {e}")
        return None


def generate_random_service():
    """Generates sample services with random data"""
    customer_key = random.choice(list(customers.keys()))
    documents = customers[customer_key]
    number = random.randint(1, 10000)

    num_tokens = random.randint(1, 2)

    # generates unique tokens
    tokens = {f"token_{random.randint(1, 999999):06d}" for _ in range(num_tokens)}

    # ensures complete uniqueness even in rare cases
    while len(tokens) < num_tokens:
        tokens.add(f"token_{random.randint(1, 999999):06d}")
        
    service = Service(
        name=f"{customer_key}_{number}",
        description=f"Service description for {customer_key}_{number}",
        tokens=list(tokens),
        customer_name=customer_key,
        documents=[]
    )

    for doc in documents:
        chunk_size = random.randint(1000, 3000)
        chunk_overlap = random.randint(200, chunk_size // 2)
        min_chunk_size = random.randint(chunk_overlap, chunk_size)

        document = Document(
            name=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )

        service.documents.append(document)

    return service


def generate_services_list_for_customer() -> list[Service]:
    """Generates a list of Service objects, one for each defined customer type."""
    all_services = []
    target_customers = list(customers.keys())

    for cust_type in target_customers:
        docs_for_cust = customers[cust_type]
        number = random.randint(100, 999)
        
        # Creation of the service for the current type
        service = Service(
            name=f"TEST_{cust_type.upper()}_{number}",
            description=f"Benchmark pipeline for {cust_type}",
            tokens=[f"tkn_{random.randint(1,999)}"],
            customer_name=cust_type,
            documents=[]
        )

        for doc_name in docs_for_cust:
            # Random chunking parameters to diversify the test
            c_size = random.randint(1000, 2000)
            document = Document(
                name=doc_name,
                chunk_size=c_size,
                chunk_overlap=int(c_size * 0.1),
                min_chunk_size=int(c_size * 0.5)
            )
            service.documents.append(document)
        
        # Add to the list of services
        all_services.append(service)
    
    return all_services


def generate_n_services(num_services: int) -> list[Service]:
    """Generates a list of random services."""
    services = []
    for _ in range(num_services):
        service = generate_random_service()
        services.append(service)
    return services


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
        return None

    except json.JSONDecodeError:
        print(f"Error: the file does not contain valid JSON: {file_path}")
        return None



def create_query_structures(num_queries: int = 5):
    queries = []
    print(f"Generating {num_queries} random queries...")

    for _ in range(num_queries):
        customer_name, query_text = get_random_question()
        service_name = get_random_service(customer_name)

        query_data = {
            "query": query_text,
            "customer_name": customer_name,
            "service_name": service_name
        }

        queries.append(query_data)

    return queries


def get_random_question():
    faqs_folder = "test/faqs/"
    faqs_files = [
        f for f in os.listdir(faqs_folder)
        if os.path.isfile(os.path.join(faqs_folder, f)) and f.endswith('.json')
    ]

    if not faqs_files:
        raise FileNotFoundError(f"No JSON file found in folder {faqs_folder}")

    faq_file = random.choice(faqs_files)
    customer_name = faq_file.removesuffix(".json")
    
    data = read_json(os.path.join(faqs_folder, faq_file))
    if not data:
        raise ValueError(f"Error reading JSON: {faq_file}")

    entry = random.choice(data)

    question = entry.get("question") # Assumes key translated or kept as original (original was 'domanda')
    if not question:
        # Check for original key if not yet translated in your JSON files
        question = entry.get("domanda")
        if not question:
            raise KeyError(f"The key 'question' or 'domanda' is missing in file {faq_file}")
    
    return customer_name, question


def get_random_service(customer_name: str):
    services_folder = f"{BASE_VECTORSTORE_PATH}/{customer_name}/"

    service_folders = [
        f for f in os.listdir(services_folder)
        if os.path.isdir(os.path.join(services_folder, f))
    ]

    if not service_folders:
        raise FileNotFoundError(f"No service folder found in '{services_folder}'")

    service_name = random.choice(service_folders)
    return service_name


def get_index_file(service_path: str) -> str:
    index_files = [
        f for f in os.listdir(service_path)
        if f.endswith(".index")
    ]

    if not index_files:
        raise FileNotFoundError(f"No .index file found in '{service_path}'")

    index_file = random.choice(index_files)
    full_index_path = os.path.join(service_path, index_file)

    print(f"Selected FAISS index: {full_index_path}")
    return full_index_path



def format_documents(documents):
    result_text = "\n---\n".join(str(doc) for doc in documents)

    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:\n{doc}\n")

    return result_text