import asyncio
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from llm.groq import groq_inference 

async def run_llm_judge():
    input_path = Path('test/results/retriever_performance_benchmark.xlsx')
    output_path = Path('test/results/retriever_performance_benchmark_judged.xlsx')

    if not input_path.exists():
        print(f"❌ Error: File {input_path} not found.")
        return

    # Loading the results benchmark
    df = pd.read_excel(input_path, sheet_name='Benchmark Details')
    
    print(f"⚖️  Starting LLM-AS-JUDGE (Model: gpt-oss-120b) on {len(df)} rows...")

    judgments = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating"):
        query = row['query']
        expected = row['query answer']
        context = f"CHUNK 1: {row['chunk1']}\nCHUNK 2: {row['chunk2']}\nCHUNK 3: {row['chunk3']}"
        
        # Judge prompt
        judge_prompt = f"""
        EVALUATE IF THE RETRIEVED CHUNKS ANSWER THE QUESTION.
        
        QUESTION: {query}
        REFERENCE CORRECT ANSWER: {expected}
        
        SYSTEM RETRIEVED CHUNKS:
        {context}
        
        RULES:
        - Respond 'OK' if at least one of the chunks contains the information to answer like the 'REFERENCE CORRECT ANSWER'.
        - Respond 'NO' if the information is missing or insufficient.
        - Respond EXCLUSIVELY with the word 'OK' or 'NO'.
        
        JUDGMENT:"""

        # LLM Call
        decision = await groq_inference(query=judge_prompt)
        
        # Cleaning to avoid LLM formatting errors
        clean_decision = "OK" if "OK" in decision.upper() else "NO"
        judgments.append(clean_decision)

    # Adding the evaluation column
    df['LLM-AS-JUDGE'] = judgments

    # Calculating accuracy statistics for each retriever
    stats = df.groupby('retriever')['LLM-AS-JUDGE'].apply(
        lambda x: f"{(x == 'OK').sum()} / {len(x)} ({(x == 'OK').mean():.2%})"
    ).reset_index()
    stats.columns = ['Retriever', 'Accuracy (OK/Total)']

    # Saving to the new Excel file
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Benchmark Details', index=False)
        stats.to_excel(writer, sheet_name='Judge Statistics', index=False)

    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 File saved at: {output_path}")

if __name__ == "__main__":
    asyncio.run(run_llm_judge())