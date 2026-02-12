import asyncio
import pandas as pd 
from pathlib import Path

# Import All Retrievers 
from src.retrievers.i_retriever import IRetriever
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

# Import classes and helpers
from test.utils import create_single_pipeline, PAGES_MAPPING, generate_services_list_for_customer


async def run_system_benchmark():
    """Runs the full system benchmark across all retrievers"""
    print(f"🚀 Starting System Benchmark...")
    print(f"⚙️  Retrievers per Service: 10\n")
    
    creation_stats = {} 
    services = generate_services_list_for_customer()

    for service in services:
        print(f"\n📦 Processing Service: {service.name.upper()}")

        retrievers = {
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
        
        # Parallel pipeline creation
        tasks = [create_single_pipeline(r, service) for r in retrievers.values()]
        results = await asyncio.gather(*tasks)
        
        service_times = {}
        for name, (pipeline, elapsed) in zip(retrievers.keys(), results):
            if pipeline is not None:
                service_times[name] = elapsed
        
        if service_times:
            creation_stats[service.name] = {
                "times": service_times,
                "service_info": service
            }

    return creation_stats 

def main():
    """Main execution point"""
    creation_stats = asyncio.run(run_system_benchmark())

    if creation_stats:
        detailed_rows = []
        
        for s_name, data in creation_stats.items():
            service_obj = data["service_info"]
            times_dict = data["times"]
            
            # Document details mapping
            doc_names = ", ".join([d.name for d in service_obj.documents])
            # Mapping pages from the utility or 'N/A' if not found
            pages_list = ", ".join([str(PAGES_MAPPING.get(d.name, "N/A")) for d in service_obj.documents])
            c_sizes = ", ".join([str(d.chunk_size) for d in service_obj.documents])
            c_overlaps = ", ".join([str(d.chunk_overlap) for d in service_obj.documents])
            m_chunks = ", ".join([str(d.min_chunk_size) for d in service_obj.documents])

            for r_name, t_val in times_dict.items():
                detailed_rows.append({
                    "Customer Type": service_obj.customer_name,
                    "Service Name": s_name,
                    "Retriever": r_name,
                    "Time (s)": round(t_val, 4),
                    "Files": doc_names,
                    "Total Pages": pages_list,
                    "Chunk Size": c_sizes,
                    "Overlap": c_overlaps,
                    "Min Chunk": m_chunks
                })
        
        df_details = pd.DataFrame(detailed_rows)

        # Aggregate statistics
        df_stats = df_details.groupby("Retriever")["Time (s)"].agg(['mean', 'min', 'max']).reset_index()
        df_stats.columns = ["Retriever", "Avg Time (s)", "Min Time (s)", "Max Time (s)"]
        df_stats = df_stats.sort_values(by="Avg Time (s)")

        # Save to Excel
        output_dir = Path('test/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = output_dir / 'pipeline_full_benchmark.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_details.to_excel(writer, sheet_name='Benchmark Details', index=False)
            df_stats.to_excel(writer, sheet_name='Retriever Statistics', index=False)
        
        print(f"\n" + "="*50)
        print(f"✅ BENCHMARK COMPLETED")
        print(f"📁 File: {excel_path}")
        print(f"📊 'Retriever Statistics' sheet with Mean, Min, and Max ready.")
        print("="*50)
    else:
        print("\n❌ No data collected. Please check the logs.")

if __name__ == '__main__':
    main()