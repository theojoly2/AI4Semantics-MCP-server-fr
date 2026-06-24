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
import resources
import tools

# Importation de l'outil de téléchargement de fichier (pour l'UI)
try:
    from file_tool import get_document_file
except ImportError:
    pass

project_dir = Path(__file__).resolve().parent
env_path = project_dir / ".env"

load_dotenv(dotenv_path=env_path)

# ====================================================================
# CRÉATION DU POOL DE THREADS
# max_workers=2 : On autorise 2 recherches SIMULTANÉES lourdes en arrière-plan.
# ====================================================================
ai_thread_pool = ThreadPoolExecutor(max_workers=2)

mcp = FastMCP(
    name="DataModellingServer",
)


# ====================================================================
# WRAPPER ASYNCHRONE POUR LA RECHERCHE
# Transforme la fonction lourde en tâche asynchrone pour ne pas bloquer le serveur
# ====================================================================
async def async_retrieve_documents(
    search_terms: str, 
    limit: int = 10, 
    return_full_document: bool = True, 
    tags: Optional[List[str]] = None
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
        tags=tags
    )

    # Exécution dans le thread pool
    results = await loop.run_in_executor(ai_thread_pool, func)

    return results


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
        tools.get_style_guide,
        name="get_style_guide",
    )
)

mcp.add_tool(
    Tool.from_function(
        async_retrieve_documents,
        name="retrieve_documents",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.upload_model,
        name="upload_model",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.add_class,
        name="add_class",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.add_attribute,
        name="add_attribute",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.add_connector,
        name="add_connector",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.metadata_checker,
        name="metadata_checker",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.reuse_check,
        name="reuse_check",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.style_guide_check,
        name="style_guide_check",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.validator_check,
        name="validator_check",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.get_available_tags,
        name="get_available_tags",
    )
)

# ====================================================================
# ENREGISTREMENT DES RESSOURCES
# ====================================================================

mcp.resource(
    "resource://model/{user}/{session_name}",
    mime_type="application/json"
)(
    resources.get_model
)

mcp.resource(
    "resource://Style_Guide}",
    mime_type="text/plain"
)(
    resources.get_style_guide
)


event_store = EventStore()
app = mcp.http_app(
    event_store=event_store,
    retry_interval=2000,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)