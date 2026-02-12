import asyncio
import json
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

# Import Other functions
from test.utils import read_json, save_results_to_excel, generate_n_services, create_single_pipeline


async def run_tests():
    """Runs tests on the retrievers"""
    print('🚀 Start testing retriever...\n')

    # Random generation of n services 
    services = generate_n_services(1)
    print(f"🛠️  Generated {len(services)} random services for testing.\n")  

    # Loading FAQs
    print("📋 Loading FAQs for testing...")
    faqs = read_json('test/faqs.json')
    
    # Creation of vectorstores for each service
    all_pipelines = {}
    creation_stats = {} # Dictionary to store timing data
    
    for service in services:
        print(f"\n📦 Creating vectorstores for: {service.name} ({service.customer_name})")
        
        # Create retrievers
        retrievers = {
            "rr": HybridRetriever(service.name, service.customer_name), 
        }
        
        # Create pipelines in parallel
        pipeline_tasks = [
            create_single_pipeline(retriever, service)
            for retriever in retrievers.values()
        ]
        
        # Execute and get results (Tuple: pipeline, time)
        pipelines_results = await asyncio.gather(*pipeline_tasks)
        
        # Map names to results
        service_pipelines = {}
        service_times = {}

        for name, (pipeline, elapsed) in zip(retrievers.keys(), pipelines_results):
            if pipeline is not None:
                service_pipelines[name] = pipeline
                service_times[name] = elapsed
        
        if service_pipelines:
            all_pipelines[service] = service_pipelines
            creation_stats[service.name] = service_times
            print(f"    ✅ {len(service_pipelines)} pipelines created for {service.name}")
        else:
            print(f"    ❌ No pipelines created for {service.name}")
    
    print(f"\n✅ All pipelines created")

    # --- CREATION TIME REPORT ---
    print("\n" + "="*50)
    print("⏱️  PIPELINE CREATION TIME REPORT")
    print("="*50)
    print(f"{'Service':<20} | {'Retriever':<10} | {'Time (s)':<10}")
    print("-" * 46)
    
    for s_name, times in creation_stats.items():
        for r_name, t_val in times.items():
            print(f"{s_name:<20} | {r_name:<10} | {t_val:.4f}s")
    print("="*50 + "\n")
    # ---------------------------------
    
    # Running tests 
    results = {}
    print("🧪 Running tests on retrievers...\n")
    
    for service, pipelines in all_pipelines.items():
        print(f"🔍 Testing Service: {service.name} ({service.customer_name})")
        
        # Filter FAQs for this customer
        service_faqs = [faq for faq in faqs if faq.get('type') == service.customer_name]
        
        if not service_faqs:
            print(f"⚠️  No FAQs found for {service.customer_name}")
            continue
        
        for pipeline_name, pipeline in pipelines.items():
            print(f"\n  📊 Retriever: {pipeline_name.upper()}")
            
            for i, faq in enumerate(service_faqs, 1):
                question = faq['question']
                expected_answer = faq.get('answer', 'N/A')
                
                print(f"\n    ❓ Question {i}: {question}")
                
                try:
                    # Execute search
                    search_results = await pipeline.search(question, top_k=3)
                    
                    if search_results and 'results' in search_results:
                        results_list = search_results['results']
                        search_time = search_results.get('search_time', 0)
                        
                        print(f"    ⏱️  Search time: {search_time:.4f}s")
                        print(f"    📄 Top {len(results_list)} results:")
                        
                        for idx, (chunk, score) in enumerate(results_list, 1):
                            preview = chunk.content[:100].replace('\n', ' ')
                            print(f"       {idx}. Score: {score:.4f} | {preview}...")
                        
                        # Save results
                        key = (service.name, service.customer_name, pipeline_name, question)
                        results[key] = {
                            'results': results_list,
                            'search_time': search_time,
                            'expected_answer': expected_answer,
                            'creation_time': creation_stats[service.name].get(pipeline_name, 0)
                        }
                    else:
                        print(f"    ⚠️  No results found")
                        
                except Exception as e:
                    print(f"    ❌ Error during search: {e}")
    
    print(f"✅ All tests completed!")
        
    return results

def main():
    """Main entry point"""
    # Run async tests
    results = asyncio.run(run_tests())

    if results:
        # Save to Excel
        excel_path = 'test/results/test_results.xlsx'
        json_path = 'test/results/test_results.json'
        
        try:
            excel_path = save_results_to_excel(results, excel_path)
            print(f"💾 Excel saved to: {excel_path}")
        except Exception as e:
            print(f"⚠️ Error saving Excel: {e}")
        
        output_file = Path(json_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in results.items():
            service_name, customer_name, pipeline_name, question = key
            results_list = []
            
            for chunk, score in value['results']:
                results_list.append({
                    'content': chunk.content,
                    'doc_id': chunk.doc_id,
                    'score': float(score)
                })
            
            serializable_results[f"{service_name}|{pipeline_name}|{question}"] = {
                'results': results_list,
                'search_time': value['search_time'],
                'creation_time': value.get('creation_time', 0),
                'expected_answer': value['expected_answer']
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON backup saved to: {output_file}")
    else:
        print("\n⚠️ No results to save")

if __name__ == '__main__':
    main()