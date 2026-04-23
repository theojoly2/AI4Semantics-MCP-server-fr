import config as cf
from typing import List

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION

def retrieve_documents(query)-> List:
    try:
        embedding = model.encode(query)
        results = client.query_points(
            collection_name=COLLECTION, 
            query=embedding.tolist(), 
            limit=3
        )
        return [(r.payload["filename"], r.payload["text"], r.score) for r in results.points]
    except Exception as e:
        print(f"✗ Search failed: {e}")
        return []

# Example usage
if __name__ == "__main__":
    try:
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' has {info.points_count} documents\n")
    except Exception as e:
        print(f"Collection info error: {e}\n")
    
    question = "how is person modeled in Core-Person vocabulary"
    results = retrieve_documents(question)

    print(type(results))
    
    if not results:
        print("\nNo results found. Try a different query.")
    else:
        for filename, text, score in results:
            print(f"\n{filename} (score: {score:.3f})\n{text[:200]}...")