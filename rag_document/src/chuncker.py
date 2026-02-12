from typing import List

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

from schemas.classes import Chunk
from src.text_processor import extract_sentences


async def semantic_chunking(
    text: str, 
    doc_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE
) -> List[Chunk]:
    """Semantic chunking based on paragraphs and sentences"""
    chunks = []
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    start_pos = 0
    chunk_id = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        sentences = await extract_sentences(para)
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) > chunk_size and current_chunk:
                if len(current_chunk) >= min_chunk_size:
                    chunk = Chunk(
                        id=f"{doc_name}_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        doc_id=doc_name,
                        start_pos=start_pos,
                        end_pos=start_pos + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                overlap_text = " ".join(current_chunk.split()[-chunk_overlap//8:])
                current_chunk = overlap_text + " " + sentence
                start_pos = start_pos + len(current_chunk) - len(overlap_text) - 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
    
    if len(current_chunk) >= min_chunk_size:
        chunk = Chunk(
            id=f"{doc_name}_chunk_{chunk_id}",
            content=current_chunk.strip(),
            doc_id=doc_name,
            start_pos=start_pos,
            end_pos=start_pos + len(current_chunk),
        )
        chunks.append(chunk)
    
    return chunks


async def semantic_chunking_v2(
    text: str, 
    doc_name: str,
    retriever_name: str, 
    doc_title: str = "", 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE
) -> List[Chunk]:
    """Semantic chunking that optimizes metadata based on the chosen retrieval"""
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk_text = ""
    start_pos = 0
    chunk_id_counter = 0
    
    for p_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para: continue
        
        sentences = await extract_sentences(para)
        
        for sentence in sentences:
            if len(current_chunk_text + " " + sentence) > chunk_size and current_chunk_text:
                if len(current_chunk_text) >= min_chunk_size:
                    
                    chunk_data = {
                        "id": f"{doc_name}_{chunk_id_counter}",
                        "content": current_chunk_text.strip(),
                        "doc_id": doc_name,
                        "start_pos": start_pos,
                        "end_pos": start_pos + len(current_chunk_text),
                        "chunk_index": chunk_id_counter 
                    }

                    if retriever_name == "ContextualHeaderRetriever":
                        chunk_data["metadata"] = {
                            "document_title": doc_title,
                            "doc_text": text[:5000] 
                        }
                    
                    elif retriever_name == "ParentDocumentRetriever":
                        chunk_data["metadata"] = {
                            "parent_id" : f"{doc_name}_p{p_idx}",
                            "parent_text": para 
                        }

                    elif retriever_name == "HierarchicalRetriever":
                        chunk_data["metadata"] = {"document_id": doc_name}

                    chunks.append(Chunk(**chunk_data))
                    chunk_id_counter += 1
                
                overlap_text = " ".join(current_chunk_text.split()[-chunk_overlap//8:])
                current_chunk_text = overlap_text + " " + sentence
                start_pos = start_pos + len(current_chunk_text) - len(overlap_text) - 1
            else:
                current_chunk_text += " " + sentence if current_chunk_text else sentence
    
    if len(current_chunk_text.strip()) >= min_chunk_size:
        
        chunk_data = {
            "id": f"{doc_name}_{chunk_id_counter}",
            "content": current_chunk_text.strip(),
            "doc_id": doc_name,
            "start_pos": start_pos,
            "end_pos": start_pos + len(current_chunk_text),
            "chunk_index": chunk_id_counter
        }

        if retriever_name == "ContextualHeaderRetriever":
            chunk_data["metadata"] = {
                "document_title": doc_title,
                "doc_text": text[:5000]
            }
        
        elif retriever_name == "ParentDocumentRetriever":
            chunk_data["metadata"] = {
                "parent_id" : f"{doc_name}_p{len(paragraphs)-1}",
                "parent_text": paragraphs[-1]
            }

        elif retriever_name == "HierarchicalRetriever":
            chunk_data["metadata"] = {"document_id": doc_name}

        chunks.append(Chunk(**chunk_data))

    return chunks


# def sliding_window(
#         text: str, 
#         doc_id: str, 
#         chunk_size: int = CHUNK_SIZE,
#         chunk_overlap: int = CHUNK_OVERLAP,
#         min_chunk_size: int = MIN_CHUNK_SIZE
# ) -> List[Chunk]:
#     """Sliding window chunking for backup"""
#     chunks = []
#     words = text.split()
    
#     for i in range(0, len(words), chunk_size - chunk_overlap):
#         chunk_words = words[i:i + chunk_size]
#         if len(chunk_words) < min_chunk_size // 6:
#             break
            
#         chunk_content = " ".join(chunk_words)
#         chunk = Chunk(
#             id=f"{doc_id}_sliding_{i//chunk_size}",
#             content=chunk_content,
#             doc_id=doc_id,
#             start_pos=i,
#             end_pos=i + len(chunk_words),
#             chunk_index=i//chunk_size
#         )
#         chunks.append(chunk)
    
#     return chunks
