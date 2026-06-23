from typing import List, Dict, Any

try:
    from .load_documents import config as cf
except ImportError:
    from load_documents import config as cf


def get_available_tags() -> Dict[str, List[Dict[str, Any]]]:
    """
    Récupère la liste de tous les tags uniques existants dans la collection Qdrant.
    Renvoie un dictionnaire {"tags": [{"tag": "nom", "count": 10}, ...]}
    """
    try:
        facets = cf.client.facet(
            collection_name=cf.COLLECTION,
            key="tags",
            limit=1000,
        )
        # On encapsule la liste dans un dictionnaire global
        tags_list = [{"tag": hit.value, "count": hit.count} for hit in facets.hits]
        return {"tags": tags_list}

    except Exception as e:
        print(f"[Erreur Serveur] Impossible de lister les tags : {e}")
        return {"tags": []}
