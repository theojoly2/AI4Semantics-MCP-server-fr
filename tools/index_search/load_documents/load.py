from pathlib import Path
from qdrant_client.models import Distance, VectorParams, PointStruct
import config as cf

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION
BATCH_SIZE = cf.BATCH_SIZE

# Create collection
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)


# Index documents
def index_documents():
    points = []
    docs_path = Path(__file__).parent / "documents"
    total_indexed = 0

    if not docs_path.exists():
        print(f"✗ Directory not found: {docs_path}")
        return

    for i, filepath in enumerate(docs_path.iterdir()):
        if filepath.is_file():
            try:
                text = filepath.read_text()
                embedding = model.encode(text)
                points.append(PointStruct(id=total_indexed + i, vector=embedding.tolist(), payload={
                    "text": text, "filename": filepath.name}))
                print(f"✓ Loaded: {filepath.name}")

                # Upload in smaller batches
                if len(points) >= BATCH_SIZE:
                    try:
                        client.upsert(collection_name=COLLECTION, points=points)
                        print(f"  → Uploaded batch of {len(points)}")
                        total_indexed += len(points)
                        points = []
                    except Exception as e:
                        print(f"  ✗ Batch upload failed: {e}")
                        points = []
            except Exception as e:
                print(f"✗ Failed: {filepath.name} - {e}")

    if points:
        try:
            client.upsert(collection_name=COLLECTION, points=points)
            print(f"  → Uploaded final batch of {len(points)}")
            total_indexed += len(points)
        except Exception as e:
            print(f"  ✗ Final batch upload failed: {e}")

    print(f"\n✓ Indexing complete: {total_indexed} documents")
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\nIndexed {len(points)} documents")


# Run indexing
index_documents()

# Example usage
if __name__ == "__main__":
    index_documents()
