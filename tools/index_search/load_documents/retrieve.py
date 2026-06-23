from pathlib import Path
import sys

from retrieve_documents import retrieve_documents
from . import config as cf

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

client = cf.client
COLLECTION = cf.COLLECTION


def get_existing_tags():
    """
    Récupère la liste de tous les tags existants dans la collection
    avec leur nombre d'occurrences. Idéal pour populer une UI.
    """
    try:
        # L'API 'facet' de Qdrant permet de récupérer les valeurs uniques d'un payload
        facets = client.facet(
            collection_name=COLLECTION,
            key="tags",
            limit=1000
        )
        # On extrait la valeur du tag ET son compteur
        # Format de sortie: [{"tag": "CNIG", "count": 12}, {"tag": "schema.data.gouv", "count": 5}]
        return [{"tag": hit.value, "count": hit.count} for hit in facets.hits]
    except Exception as e:
        print(f"Impossible de lister les tags : {e}")
        return []


if __name__ == "__main__":
    try:
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' has {info.points_count} documents\n")

        # --- 1. Récupération des tags pour l'Interface Graphique ---
        tags_en_base = get_existing_tags()
        print("Tags disponibles pour l'interface utilisateur :")
        for item in tags_en_base:
            print(f"- {item['tag']} ({item['count']} documents)")
        print("\n")

    except Exception as e:
        print(f"Collection info error: {e}\n")

    question = "existe t il des standards sur la faible emission?"

    # --- 2. Définir les tags que l'on veut filtrer ---
    # Ici on simule ce que l'utilisateur a coché dans l'interface
    tags_selectionnes_par_user = ["CNIG", "schema.data.gouv"]

    # --- 3. Passer les tags à votre fonction de recherche ---
    results = retrieve_documents(question, limit=3, tags=tags_selectionnes_par_user)

    if not results:
        print("\nNo results found. Try a different query.")
    else:
        for filename, text, score in results:
            print(f"\n{filename} (score: {score:.3f})\n{text[:200]}...")
