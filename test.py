from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding simple strings...")
embeddings = model.encode(["This is a test.", "This is another test."])

print(f"Success! Embeddings shape: {embeddings.shape}")