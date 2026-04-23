from typing import Any, Literal
from os import getenv
from json import loads
from requests import post, Response
from pandas import DataFrame, read_excel
from pathlib import Path


from dotenv import load_dotenv
load_dotenv(dotenv_path=Path.cwd() / ".env")

BASE_PATH: str = 'tools/index_search'
BASE_URL: str = str(getenv('AZURE_SEARCH_SERVICE_ENDPOINT'))
API_VERSION: str = str(getenv('AZURE_SEARCH_API_VERSION'))
INDEX_NAME: str = str(getenv('AZURE_SEARCH_INDEX_NAME'))
API_KEY: str = str(getenv('AZURE_SEARCH_ADMIN_KEY'))
SEMANTIC_CONFIG: str = str(getenv('AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG'))
URL = f'{BASE_URL}/indexes/{INDEX_NAME}/docs/search?api-version={API_VERSION}'
CREATE_KWARGS: dict[str, Any] = {
    'temperature': 0,
    'stream': False,
}

Vocabulary = Literal[
    "CAV",
    "CBV",
    "CCCEV",
    "CLV",
    "CPEV",
    "CPOV",
    "CPSV",
    "CPV",
    "OSLO_Observaties_en_Metingen",
    "OSLO_Organisatie",
    "OSLO_Persoon",
    "OSLO_Adres",
    "OSLO_Gebouw",
]

df: DataFrame = read_excel(f'{BASE_PATH}/vocabularies.xlsx')

VOCABULARIES: list[Vocabulary] = df['NAME'].values.tolist()
DESCRIPTIONS: dict[str, str] = {
    df['NAME'].iloc[i]: df['DESCRIPTION'].iloc[i]
    for i
    in range(len(df))
}


def _search_index(
    search_term: str,
    backend: str = "azure",
    **query_kwargs: Any,
) -> dict[str, Any]:
    """
    backend: 'azure' or 'elasticsearch'
    """
    if backend == "azure":
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "api-key": API_KEY
        }
        query: dict[str, Any] = {
            'queryType': 'semantic',
            'semanticConfiguration': SEMANTIC_CONFIG,
            'search': search_term,
            'count': True,
            'searchFields': 'Content, Keyword',
            'top': 5,
        }
        query.update(query_kwargs)
        response: Response = post(
            URL,
            headers=headers,
            json=query,
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {}
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


async def retrieve_documents(
    search_terms: str,
    vocabularies: list[Vocabulary] = [],
    number_of_documents: int = 5,
    backend: str = "azure",
) -> dict[str, dict[str, Any]]:
    if not vocabularies:
        filter: str = ''
    else:
        filter = ' or '.join(
            f"search.ismatch('{voc}*', 'Title')"
            for voc in vocabularies if voc in VOCABULARIES
        )
    search_res = _search_index(
        search_terms,
        filter=filter,
        top=number_of_documents,
        backend=backend,
    )
    docs: dict[str, dict[str, str]] = {
        f"(doc{i+1})": {
            k: v
            for k, v in doc.items()
            if k not in [
                '@search.score',
                '@search.rerankerScore',
            ]
        }
        for i, doc in enumerate(search_res.get('value', []))
    }
    return docs


descriptions = "\n\n".join(
    f"[{name}]:" + "\n" + description
    for name, description
    in DESCRIPTIONS.items()
)

retrieve_documents.__doc__ = f"""
    Provides context regarding the search_term by retrieving
    relevant documents from a knowledge database.

    Args:
        search_terms (str):
            Keywords summarizing the user's input or
            a short summary of the user's input

        vocabularies (list[
            Literal[
                {VOCABULARIES}
            ]
        ]):
            The vocabularies most relevant to the user's input.
            Defaults to [] (if none of the vocabularies are relevant).

            The relevancy is based on the following descriptions:
            {descriptions}

        n_documents (int):
            The maximum number of documents retrieved.
            Defaults to 5.

    Returns:
        dict[str, dict[str, str]]:
            The documents, in the format (doc<i>): <document>,
            belonging to the *vocabularies*, and relevant to *search_terms*.
            The documents are in JSON format as well.
    """
