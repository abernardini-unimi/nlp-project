import asyncio
import pandas as pd
from pathlib import Path

# Retriever Imports
from src.retrievers.contextualHeader.chr import ContextualHeaderRetriever 
from src.retrievers.hierarchicalIndices.hi import HierarchicalRetriever 
from src.retrievers.hybrid.hybrid_retriever import HybridRetriever
from src.retrievers.multiQueryRAG.mqr import MultiQueryRetriever
from src.retrievers.parentDocument.pdr import ParentDocumentRetriever
from src.retrievers.queryTransformations.qt import QueryTransformRetriever 
from src.retrievers.relevantSegmentExtraction.rse import RelevantSegmentRetriever
from src.retrievers.reranking.rr import RerankingRetriever
from src.retrievers.semantic.semantic_retriever import SemanticRetriever
from src.retrievers.sparse.bm25_retriever import Bm25Retriever

# Classes and Utils Imports
from test.utils import read_json, create_single_pipeline, generate_services_list_for_customer, PAGES_MAPPING


async def run_full_benchmark_test():
    """Runs a comprehensive benchmark of the retrieval results"""
    print("🚀 Starting Full Performance Benchmark...")
    
    # Load FAQs
    faqs = read_json('test/faqs.json')
    if not faqs:
        print("❌ Error: file test/faqs.json not found or invalid.")
        return

    # Generate services list (one for each customer)
    services_to_test = generate_services_list_for_customer()
    all_results_data = []

    for service in services_to_test:
        cust_type = service.customer_name
        print(f"\n📦 Processing Customer: {cust_type.upper()}")
        
        # Filter FAQs for this specific customer
        service_faqs = [f for f in faqs if f.get('type') == cust_type]
        if not service_faqs:
            print(f"  ⚠️ No FAQs found for {cust_type}, skipping tests.")
            continue

        # Initialize Retrievers
        retrievers_map = {
            "ContextualHeaderRetriever": ContextualHeaderRetriever(service.name, service.customer_name),
            "HierarchicalRetriever": HierarchicalRetriever(service.name, service.customer_name),
            "HybridRetriever": HybridRetriever(service.name, service.customer_name),
            "MultiQueryRetriever": MultiQueryRetriever(service.name, service.customer_name),
            "ParentDocumentRetriever": ParentDocumentRetriever(service.name, service.customer_name),
            "QueryTransformRetriever": QueryTransformRetriever(service.name, service.customer_name),
            "RelevantSegmentRetriever": RelevantSegmentRetriever(service.name, service.customer_name),
            "RerankingRetriever": RerankingRetriever(service.name, service.customer_name),
            "SemanticRetriever": SemanticRetriever(service.name, service.customer_name),
            "Bm25Retriever": Bm25Retriever(service.name, service.customer_name)
        }

        # 1. Pipeline Creation (Batch)
        print(f"  🛠️  Creating 10 pipelines for {service.name}...")
        tasks = [create_single_pipeline(r, service) for r in retrievers_map.values()]
        created_pipelines = await asyncio.gather(*tasks)
        
        pipelines_ready = {
            name: pipe for name, (pipe, creation_time) in zip(retrievers_map.keys(), created_pipelines) 
            if pipe is not None
        }

        # 2. Execute Search Tests
        for r_name, pipeline in pipelines_ready.items():
            print(f"  🔍 Executing queries for {r_name}...")
            for faq in service_faqs:
                query = faq['question']
                expected_answer = faq.get('answer', 'N/A')
                
                try:
                    # Run search (top_k=3 to get the 3 required chunks)
                    search_res = await pipeline.search(query, top_k=3)
                    results_list = search_res.get('results', [])
                    search_time = search_res.get('search_time', 0)

                    # Extract content from the first 3 chunks
                    chunks = []
                    for i in range(3):
                        if i < len(results_list):
                            # Extract clean chunk content
                            content = results_list[i][0].content.replace("\n", " ")
                            chunks.append(content)
                        else:
                            chunks.append("N/A")

                    all_results_data.append({
                        "customer": cust_type,
                        "service": service.name,
                        "retriever": r_name,
                        "query": query,
                        "query answer": expected_answer,
                        "chunk1": chunks[0],
                        "chunk2": chunks[1],
                        "chunk3": chunks[2],
                        "search time": round(search_time, 4)
                    })
                except Exception as e:
                    print(f"    ❌ Error during search ({r_name}): {e}")

    return all_results_data

def main():
    """Main entry point"""
    # Execute the test
    results = asyncio.run(run_full_benchmark_test())

    if results:
        # 1. Results Sheet (Main)
        df_main = pd.DataFrame(results)

        # 2. Details/Stats Sheet (Aggregated by Retriever)
        df_stats = df_main.groupby("retriever")["search time"].agg(['mean', 'min', 'max']).reset_index()
        df_stats.columns = ["Retriever", "Avg Time (s)", "Min Time (s)", "Max Time (s)"]
        df_stats = df_stats.sort_values(by="Avg Time (s)", ascending=True)

        # Setup save path
        output_dir = Path('test/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = output_dir / 'retriever_performance_benchmark.xlsx'

        # Write to Excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='Benchmark Details', index=False)
            df_stats.to_excel(writer, sheet_name='Retriever Stats', index=False)

        print(f"\n" + "="*50)
        print(f"✅ BENCHMARK COMPLETED SUCCESSFULLY")
        print(f"📊 File generated: {excel_path}")
        print(f"📈 Processed {len(df_main)} total queries.")
        print("="*50)
    else:
        print("\n❌ No data collected. Please check input files (faqs.json) and logs.")

if __name__ == '__main__':
    main()