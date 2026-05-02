from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
      
    def embed_chunks(self, chunks):
        clean_texts = []
        
        for chunk in chunks:
            content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
            if content is None:
                continue
                
            # 1. Force standard string and strip whitespace
            text_str = str(content).strip()
          
            text_str = text_str.encode('utf-8', 'ignore').decode('utf-8')
            
            if text_str: 
                clean_texts.append(text_str)
                
        print(f"[INFO] Purged PDF corruption. Batch encoding {len(clean_texts)} clean chunks...")
        
        # 3. Back to fast batch processing!
        embeddings = self.model.encode(clean_texts, show_progress_bar=True)
        
        return embeddings
        

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)