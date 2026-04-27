from pathlib import Path
from hashlib import sha256
from qdrant_client.models import Distance, VectorParams, PointStruct
import config as cf


client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION
BATCH_SIZE = cf.BATCH_SIZE


def setup_collection():
    """Create collection or recreate if exists."""
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION for c in collections)
    
    if collection_exists:
        print(f"⚠ Collection '{COLLECTION}' already exists.")
        choice = input("  Delete and recreate? (y/n): ").strip().lower()
        
        if choice == 'y':
            client.delete_collection(collection_name=COLLECTION)
            print(f"✓ Deleted collection '{COLLECTION}'")
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"✓ Created new collection '{COLLECTION}'")
            return True  # Fresh collection
        else:
            print(f"✓ Using existing collection '{COLLECTION}'")
            return False  # Existing collection
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"✓ Created collection '{COLLECTION}'")
        return True  # Fresh collection


def get_existing_ids():
    """Retrieve all existing point IDs from the collection."""
    try:
        result = client.scroll(
            collection_name=COLLECTION,
            limit=10000,  # Adjust if you have more documents
            with_payload=False,
            with_vectors=False
        )
        existing_ids = {point.id for point in result[0]}
        print(f"  → Found {len(existing_ids)} existing documents")
        return existing_ids
    except Exception as e:
        print(f"  ✗ Failed to retrieve existing IDs: {e}")
        return set()


def generate_stable_id(text: str) -> int:
    """Generate a stable numeric ID from document content using SHA-256 hash."""
    hash_bytes = sha256(text.encode('utf-8')).digest()
    # Convert first 8 bytes to int (positive 64-bit integer)
    return int.from_bytes(hash_bytes[:8], byteorder='big') & 0x7FFFFFFFFFFFFFFF


def index_documents():
    """Index documents with duplicate detection."""
    is_fresh = setup_collection()
    
    # Get existing IDs only if we're adding to an existing collection
    existing_ids = set() if is_fresh else get_existing_ids()
    
    points = []
    docs_path = Path(__file__).parent / "documents"
    total_indexed = 0
    total_skipped = 0

    if not docs_path.exists():
        print(f"✗ Directory not found: {docs_path}")
        return

    print(f"\n📂 Scanning documents in: {docs_path}")
    
    for filepath in docs_path.iterdir():
        if filepath.is_file():
            try:
                text = filepath.read_text(encoding='utf-8')
                doc_id = generate_stable_id(text)
                
                # Skip if already indexed (only when using existing collection)
                if not is_fresh and doc_id in existing_ids:
                    print(f"⊘ Skipped (duplicate): {filepath.name}")
                    total_skipped += 1
                    continue
                
                embedding = model.encode(text)
                points.append(PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload={"text": text, "filename": filepath.name}
                ))
                print(f"✓ Loaded: {filepath.name}")

                # Upload in batches
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
                print(f"✗ Failed to read {filepath.name}: {e}")

    # Upload final batch
    if points:
        try:
            client.upsert(collection_name=COLLECTION, points=points)
            print(f"  → Uploaded final batch of {len(points)}")
            total_indexed += len(points)
        except Exception as e:
            print(f"  ✗ Final batch upload failed: {e}")

    print(f"\n{'='*50}")
    print(f"✓ Indexing complete")
    print(f"  • Documents indexed: {total_indexed}")
    if total_skipped > 0:
        print(f"  • Documents skipped (duplicates): {total_skipped}")
    print(f"{'='*50}")


if __name__ == "__main__":
    index_documents()