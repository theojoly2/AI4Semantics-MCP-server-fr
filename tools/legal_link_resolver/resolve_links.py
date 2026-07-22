"""
GlossaryAI Legal & RDF Link Resolver

Détecte automatiquement dans les chunks retournés :
- les références à des articles juridiques ("article 5", "article L321-4"...)
- les liens RDF/SKOS (broader, narrower, related)

Puis relance des recherches ciblées dans les documents indexés
(1 saut de profondeur par défaut, anti-boucle).
"""

from typing import List, Dict, Any, Optional
import re
from pathlib import Path

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

try:
    from ..index_search import retrieve_documents
except ImportError:
    from tools.index_search import retrieve_documents

try:
    from ..index_search.load_documents import config as cf
except ImportError:
    from tools.index_search.load_documents import config as cf


RE_ARTICLE_MENTION = re.compile(
    r"\b(?:article|art\.?)\s+([LRD]?\d+[\d\-\.]*(?:\s*-\s*\d+)?)\b",
    re.IGNORECASE,
)

RE_GENERIC_ARTICLE = re.compile(
    r"\b(?:article|art\.?)\s+([A-Za-z0-9][A-Za-z0-9\-\.]*(?:\s*-\s*\d+)?)\b",
    re.IGNORECASE,
)

CONFIG = getattr(cf, "config", {}) if hasattr(cf, "config") else {}
LINK_RESOLVER_CFG = CONFIG.get("link_resolver", {})
DEFAULT_MAX_DEPTH = int(LINK_RESOLVER_CFG.get("max_depth", 1))
DEFAULT_MAX_LINKS = int(LINK_RESOLVER_CFG.get("max_links_per_step", 5))
DEFAULT_MIN_SCORE = float(LINK_RESOLVER_CFG.get("min_score_threshold", 0.5))


def _normalize_article_ref(raw: str) -> str:
    """Normalise une référence d'article pour matching."""
    return re.sub(r"\s+", "", raw.upper()).replace("ART.", "").replace("ARTICLE", "")


def _extract_cited_articles(text: str) -> List[str]:
    refs = set()
    for m in RE_ARTICLE_MENTION.finditer(text):
        refs.add(_normalize_article_ref(m.group(1)))
    return sorted(refs)


def _extract_rdf_links(chunk: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extrait les liens RDF/SKOS d'un chunk."""
    links = []
    for relation in ("broader", "narrower", "related"):
        for uri in chunk.get(relation, []) or []:
            links.append({"type": relation, "uri": str(uri)})
    return links


def _extract_document_filter_for_article(
    article_ref: str,
    source_document_name: str,
    all_document_names: List[str],
) -> Optional[str]:
    """
    Essaye de trouver dans quel document chercher l'article cité.
    Par défaut, on reste dans le même document.
    """
    normalized = _normalize_article_ref(article_ref)
    # Si la source est un code/règlement, privilégier la source elle-même
    if source_document_name:
        return source_document_name
    return None


def _deduplicate_chunks(chunks: List[Dict[str, Any]], seen_keys: set) -> List[Dict[str, Any]]:
    out = []
    for c in chunks:
        key = (
            c.get("document_id", ""),
            c.get("chunk_index", 0),
            c.get("article", ""),
            c.get("concept_uri", ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(c)
    return out


def resolve_links(
    chunks: List[Dict[str, Any]],
    max_depth: int = DEFAULT_MAX_DEPTH,
    visited: Optional[List[str]] = None,
    current_depth: int = 0,
) -> Dict[str, Any]:
    """
    Résout automatiquement les liens juridiques et RDF présents dans les chunks.

    Retourne :
    {
        "expanded_chunks": [chunks originaux + chunks liés],
        "links_found": [{"type": "article", "ref": "5", "source": "...", "query": "..."}, ...],
        "depth_reached": int,
        "queries_executed": int,
    }
    """
    if not chunks:
        return {
            "expanded_chunks": [],
            "links_found": [],
            "depth_reached": current_depth,
            "queries_executed": 0,
        }

    if visited is None:
        visited = []

    if current_depth >= max_depth:
        return {
            "expanded_chunks": chunks,
            "links_found": [],
            "depth_reached": current_depth,
            "queries_executed": 0,
        }

    links_found: List[Dict[str, Any]] = []
    extra_queries: List[Dict[str, Any]] = []
    visited_set = set(str(v) for v in visited)

    all_document_names = sorted({str(c.get("document_name", "") or c.get("filename", "")) for c in chunks})

    for chunk in chunks:
        source_doc = str(chunk.get("document_name", "") or chunk.get("filename", ""))

        # Liens juridiques
        text = chunk.get("text", "")
        cited = _extract_cited_articles(text)
        for ref in cited:
            norm = _normalize_article_ref(ref)
            visited_key = f"article:{source_doc}:{norm}"
            if visited_key in visited_set:
                continue
            visited_set.add(visited_key)

            # Requête ciblée : "article 5" + nom du document
            query = f"article {ref}"
            document_filter = _extract_document_filter_for_article(ref, source_doc, all_document_names)
            links_found.append({
                "type": "article",
                "ref": ref,
                "source": source_doc,
                "query": query,
                "document_filter": document_filter,
            })
            extra_queries.append({
                "query": query,
                "document_filter": document_filter,
                "reason": f"Article {ref} cité dans {source_doc}",
            })

        # Liens RDF
        rdf_links = _extract_rdf_links(chunk)
        for link in rdf_links:
            uri = link["uri"]
            visited_key = f"rdf:{uri}"
            if visited_key in visited_set:
                continue
            visited_set.add(visited_key)
            links_found.append({
                "type": f"rdf_{link['type']}",
                "uri": uri,
                "source": source_doc,
                "query": uri,
            })
            extra_queries.append({
                "query": uri,
                "document_filter": None,
                "reason": f"Concept lié ({link['type']}) : {uri}",
            })

    if not extra_queries:
        return {
            "expanded_chunks": chunks,
            "links_found": [],
            "depth_reached": current_depth,
            "queries_executed": 0,
        }

    # Limiter le nombre de liens explorés par étape
    extra_queries = extra_queries[:DEFAULT_MAX_LINKS]
    links_found = links_found[:DEFAULT_MAX_LINKS]

    resolved_chunks: List[Dict[str, Any]] = []
    queries_executed = 0

    for q in extra_queries:
        try:
            results = retrieve_documents(
                search_terms=q["query"],
                limit=3,
                return_full_document=False,
                document_filter=q.get("document_filter"),
            )
            queries_executed += 1
            for r in results:
                # On garde le chunk détaillé
                chunk = r.get("chunk", _payload_to_chunk_dict(r))
                if chunk.get("score", 0.0) >= DEFAULT_MIN_SCORE:
                    chunk["resolved_from"] = q["reason"]
                    resolved_chunks.append(chunk)
        except Exception as e:
            print(f"[!] resolve_links query failed ({q}): {e}")
            continue

    # Dédoublonner
    seen_keys = set()
    all_expanded = _deduplicate_chunks(chunks + resolved_chunks, seen_keys)

    # Appel récursif si on n'a pas atteint la profondeur max et si on a trouvé des liens
    if current_depth + 1 < max_depth and resolved_chunks:
        deeper = resolve_links(
            chunks=all_expanded,
            max_depth=max_depth,
            visited=list(visited_set),
            current_depth=current_depth + 1,
        )
        return {
            "expanded_chunks": deeper["expanded_chunks"],
            "links_found": links_found + deeper["links_found"],
            "depth_reached": deeper["depth_reached"],
            "queries_executed": queries_executed + deeper["queries_executed"],
        }

    return {
        "expanded_chunks": all_expanded,
        "links_found": links_found,
        "depth_reached": current_depth + 1,
        "queries_executed": queries_executed,
    }


def _payload_to_chunk_dict(payload: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
    return {
        "text": str(payload.get("text", "")),
        "filename": str(payload.get("filename", "")),
        "document_name": str(payload.get("document_name", "")),
        "document_id": str(payload.get("document_id", "")),
        "chunk_index": int(payload.get("chunk_index", 0)),
        "chunk_count": int(payload.get("chunk_count", 1)),
        "source_path": str(payload.get("source_path", "")),
        "source_extension": str(payload.get("source_extension", "")),
        "doctype": str(payload.get("doctype", "generic")),
        "tags": payload.get("tags", []) or [],
        "term": str(payload.get("term", "")),
        "definition": str(payload.get("definition", "")),
        "synonyms": payload.get("synonyms", []) or [],
        "article": str(payload.get("article", "")),
        "cited_articles": payload.get("cited_articles", []) or [],
        "concept_uri": str(payload.get("concept_uri", "")),
        "broader": payload.get("broader", []) or [],
        "narrower": payload.get("narrower", []) or [],
        "related": payload.get("related", []) or [],
        "section_title": str(payload.get("section_title", "")),
        "doc_summary": str(payload.get("doc_summary", "")),
        "score": float(payload.get("score", score)),
    }
