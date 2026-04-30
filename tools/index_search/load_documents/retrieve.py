from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from retrieve_documents import retrieve_documents
from . import config as cf


client = cf.client
COLLECTION = cf.COLLECTION


if __name__ == "__main__":
    try:
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' has {info.points_count} documents\n")
    except Exception as e:
        print(f"Collection info error: {e}\n")

    question = "existe t il des standards sur la faible emission?"
    results = retrieve_documents(question, limit=3)

    if not results:
        print("\nNo results found. Try a different query.")
    else:
        for filename, text, score in results:
            print(f"\n{filename} (score: {score:.3f})\n{text[:200]}...")
