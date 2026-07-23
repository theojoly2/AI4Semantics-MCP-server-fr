import os
# Désactive le parallélisme interne du Tokenizer HuggingFace pour éviter les deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
from typing import List, Optional, Any

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.server.event_store import EventStore
import tools

project_dir = Path(__file__).resolve().parent
env_path = project_dir / ".env"

load_dotenv(dotenv_path=env_path)

# ====================================================================
# CRÉATION DU POOL DE THREADS
# max_workers=2 : On autorise 2 recherches SIMULTANÉES lourdes en arrière-plan.
# ====================================================================
ai_thread_pool = ThreadPoolExecutor(max_workers=2)

mcp = FastMCP(
    name="GlossaryAI",
)


# ====================================================================
# WRAPPER ASYNCHRONE POUR LA RECHERCHE
# Transforme la fonction lourde en tâche asynchrone pour ne pas bloquer le serveur
# ====================================================================
async def async_retrieve_documents(
    search_terms: str,
    limit: int = 10,
    return_full_document: bool = True,
    tags: Optional[List[str]] = None,
) -> list:
    """
    Recherche des documents ou standards dans la base de connaissances.
    """
    loop = asyncio.get_running_loop()

    # Prépare la fonction avec ses arguments
    func = functools.partial(
        tools.retrieve_documents,
        search_terms=search_terms,
        limit=limit,
        return_full_document=return_full_document,
        tags=tags,
    )

    # Exécution dans le thread pool
    results = await loop.run_in_executor(ai_thread_pool, func)

    return results


async def async_resolve_links(
    chunks: List[dict],
    max_depth: int = 1,
    visited: Optional[List[str]] = None,
) -> List[dict]:
    """
    Détecte les liens juridiques et conceptuels dans les chunks et remonte
    les chunks liés (1 saut de profondeur par défaut, anti-boucle).
    """
    loop = asyncio.get_running_loop()
    func = functools.partial(
        tools.resolve_links,
        chunks=chunks,
        max_depth=max_depth,
        visited=visited,
    )
    return await loop.run_in_executor(ai_thread_pool, func)


async def async_compare_concepts(
    terms: List[str],
    limit: int = 5,
) -> dict:
    """
    Compare plusieurs termes et propose une définition convergente avec sources.
    """
    loop = asyncio.get_running_loop()
    func = functools.partial(
        tools.compare_concepts,
        terms=terms,
        limit=limit,
    )
    return await loop.run_in_executor(ai_thread_pool, func)


# ====================================================================
# ENREGISTREMENT DES OUTILS
# ====================================================================

mcp.add_tool(
    Tool.from_function(
        tools.plan_workflow_with_tools,
        name="plan_workflow_with_tools",
    ),
)

mcp.add_tool(
    Tool.from_function(
        async_retrieve_documents,
        name="retrieve_documents",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.get_available_tags,
        name="get_available_tags",
    )
)

mcp.add_tool(
    Tool.from_function(
        async_resolve_links,
        name="resolve_links",
    )
)

mcp.add_tool(
    Tool.from_function(
        async_compare_concepts,
        name="compare_concepts",
    )
)


event_store = EventStore()
app = mcp.http_app(
    event_store=event_store,
    retry_interval=2000,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
