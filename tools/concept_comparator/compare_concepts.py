"""
GlossaryAI Concept Comparator

Compare plusieurs termes en les recherchant dans la base vectorielle,
puis demande à un LLM de proposer une définition convergente/terme canonique.
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from pathlib import Path

try:
    from ..index_search import retrieve_documents
except ImportError:
    from tools.index_search import retrieve_documents

try:
    from ..index_search.load_documents import config as cf
except ImportError:
    from tools.index_search.load_documents import config as cf


_ROOT_ENV = Path(__file__).resolve().parent
while _ROOT_ENV.parent != _ROOT_ENV:
    if (_ROOT_ENV / ".env").exists():
        break
    _ROOT_ENV = _ROOT_ENV.parent

from dotenv import load_dotenv
load_dotenv(_ROOT_ENV / ".env")

_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
_URL_API = os.getenv("URL_LLM_API", "")
_llm_client = OpenAI(base_url=_URL_API, api_key=_LLM_API_KEY)
_LLM_MODEL = os.getenv("LLM_MODEL", "")

CONFIG = getattr(cf, "config", {}) if hasattr(cf, "config") else {}
COMPARATOR_CFG = CONFIG.get("concept_comparator", {})
DEFAULT_LIMIT = int(COMPARATOR_CFG.get("max_results_per_term", 5))
MAX_TERMS = int(COMPARATOR_CFG.get("max_terms", 5))


_COMPARISON_SYSTEM_PROMPT = (
    "Tu es un assistant expert en terminologie et en alignement de vocabulaires. "
    "On te fournit plusieurs termes et, pour chacun, des définitions/issues de sources différentes. "
    "Ta mission est de :\n"
    "1. Comparer les définitions et identifier les points communs et les divergences.\n"
    "2. Proposer un terme canonique (le plus neutre et reconnu possible).\n"
    "3. Proposer une définition convergente unique qui synthétise les sources sans les trahir.\n"
    "4. Indiquer clairement les sources utilisées (document, article, concept).\n"
    "5. Si les termes ne recouvrent pas exactement le même sens, le dire explicitement.\n"
    "N'invente rien. Base-toi uniquement sur les extraits fournis."
)


def _chunk_key(chunk: Dict[str, Any]) -> str:
    return f"{chunk.get('document_id', '')}:{chunk.get('chunk_index', 0)}:{chunk.get('article', '')}:{chunk.get('concept_uri', '')}"


def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for c in chunks:
        key = _chunk_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def compare_concepts(
    terms: List[str],
    limit: int = DEFAULT_LIMIT,
) -> Dict[str, Any]:
    """
    Compare une liste de termes et retourne une synthèse convergente.

    Retourne :
    {
        "terms": ["terme1", "terme2", ...],
        "sources": [chunks utilisés],
        "synthesis": "réponse du LLM",
        "queries_executed": int,
    }
    """
    if not terms:
        return {
            "terms": [],
            "sources": [],
            "synthesis": "Aucun terme fourni pour la comparaison.",
            "queries_executed": 0,
        }

    terms = terms[:MAX_TERMS]

    all_chunks: List[Dict[str, Any]] = []
    queries_executed = 0

    for term in terms:
        try:
            results = retrieve_documents(
                search_terms=term,
                limit=limit,
                return_full_document=False,
            )
            queries_executed += 1
            for r in results:
                chunk = r.get("chunk")
                if not chunk:
                    chunk = {
                        "text": r.get("text", ""),
                        "filename": r.get("filename", ""),
                        "document_name": r.get("document_name", ""),
                        "document_id": r.get("document_id", ""),
                        "chunk_index": r.get("best_chunk_index", 0),
                        "score": r.get("score", 0.0),
                        "article": "",
                        "concept_uri": "",
                        "term": "",
                    }
                chunk["matched_term"] = term
                all_chunks.append(chunk)
        except Exception as e:
            print(f"[!] compare_concepts failed for term '{term}': {e}")
            continue

    all_chunks = _deduplicate_chunks(all_chunks)

    if not all_chunks:
        return {
            "terms": terms,
            "sources": [],
            "synthesis": "Aucune source trouvée pour les termes fournis.",
            "queries_executed": queries_executed,
        }

    # Construction du prompt utilisateur
    sections = []
    for i, chunk in enumerate(all_chunks, 1):
        header = f"Source {i}"
        if chunk.get("term"):
            header += f" | Terme indexé : {chunk['term']}"
        if chunk.get("article"):
            header += f" | Article : {chunk['article']}"
        if chunk.get("concept_uri"):
            header += f" | Concept URI : {chunk['concept_uri']}"
        header += f" | Document : {chunk.get('filename', chunk.get('document_name', ''))}"
        sections.append(f"{header}\n{chunk.get('text', '')}\n")

    user_message = (
        f"Termes à comparer : {', '.join(terms)}\n\n"
        "Extraits de sources :\n\n"
        + "\n---\n".join(sections)
        + "\n\nPropose un terme canonique et une définition convergente. Cite les sources."
    )

    synthesis = ""
    try:
        response = _llm_client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _COMPARISON_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        synthesis = response.choices[0].message.content or ""
        synthesis = synthesis.strip()
    except Exception as e:
        print(f"[!] compare_concepts LLM synthesis failed: {e}")
        synthesis = "Erreur lors de la synthèse par le LLM."

    return {
        "terms": terms,
        "sources": all_chunks,
        "synthesis": synthesis,
        "queries_executed": queries_executed,
    }
