from .load_documents import config as cf
from typing import List

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION

def retrieve_documents(search_terms: str)-> List:
    try:
        embedding = model.encode(search_terms)
        results = client.query_points(
            collection_name=COLLECTION, 
            query=embedding.tolist(), 
            limit=3
        )
        return [(r.payload["filename"], r.payload["text"], r.score) for r in results.points]
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


retrieve_documents.__doc__ = """
    Provides context regarding the query by retrieving
    relevant documents from a knowledge database.

    Args:
        search_terms (str):
            Keywords summarizing the user's input or
            a short summary of the user's input

    Returns:
        List[Tuple[str, str, float]]:
            A list of tuples containing (filename, text, score) for each
            retrieved document, with a maximum of 3 documents.
    """
