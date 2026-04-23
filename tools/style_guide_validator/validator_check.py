from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from resources.semantic_model.utils import get_model
from fastmcp import Context

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from rdflib import Graph, URIRef
import re
from config import load_config

config = load_config()

STYLE_GYDE_XLS_PATH = Path(config["file_paths"]["style_guide_xls"])

df = pd.read_excel(STYLE_GYDE_XLS_PATH, index_col="Rule")
RULES_DICT = df.to_dict(orient="index")

SHACL_NS = "http://www.w3.org/ns/shacl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def parse_validator_output(response: Any) -> dict[str, Any]:
    """
    Parse SHACL validator outputs in either JSON-LD (current) or RDF/XML (legacy).
    Supports:
      - requests.Response (uses .text and .headers)
      - raw JSON/XML strings
      - Python dict already loaded from JSON

    Returns:
      errors: dict with keys:
        - "Success": list with success message (if conforms true)
        - "<message>": list of concept IRIs for that message
        - "other": dict of message -> list of raw payloads for blank nodes or missing concept IRIs
        - "Parsing": string with raw content when parsing fails
    """
    errors: dict[str, Any] = {"other": {}}

    # ---- Get the raw text/body and content-type, from various kinds of inputs ----
    text = None
    content_type = ""
    if hasattr(response, "text"):  # requests.Response
        text = response.text
        try:
            content_type = (getattr(response, "headers", {}) or {}).get("Content-Type", "") or ""
        except Exception:
            content_type = ""
    elif isinstance(response, (str, bytes)):
        text = response.decode("utf-8", errors="replace") if isinstance(response, bytes) else response
    elif isinstance(response, dict):
        # Already parsed JSON
        return _parse_jsonld_obj(response, errors)
    else:
        errors["Parsing"] = "Unsupported response type."
        return errors

    # ---- Try JSON first if content-type suggests it or the payload looks like JSON ----
    looks_like_json = text.lstrip().startswith("{") or "json" in content_type.lower()
    if looks_like_json:
        try:
            obj = json.loads(text)
            return _parse_jsonld_obj(obj, errors)
        except Exception:
            # Fall through to XML attempt
            pass

    # ---- Attempt XML parse (legacy RDF/XML) ----
    try:
        root = ET.fromstring(text)

        # Find conforms flag
        conforms = root.find(f'.//{{{SHACL_NS}}}conforms')
        if conforms is not None and (conforms.text or "").strip().lower() == "true":
            errors["Success"] = ["Validation successful: No errors in the taxonomy"]
            return errors

        # Iterate ValidationResult nodes
        for result in root.findall(f'.//{{{SHACL_NS}}}ValidationResult'):
            message = result.find(f'{{{SHACL_NS}}}resultMessage')
            focus_node = result.find(f'{{{SHACL_NS}}}focusNode')

            if message is None:
                # Gracefully skip if no message
                info_xml = ET.tostring(result, encoding='unicode')
                errors["other"].setdefault("Unknown message", []).append(info_xml)
                continue

            msg_text = (message.text or "").strip()
            concept = None

            # In RDF/XML, focusNode can use rdf:resource (IRI) or rdf:nodeID (blank node)
            if focus_node is not None:
                iri = focus_node.attrib.get(f'{{{RDF_NS}}}resource')
                bnode = focus_node.attrib.get(f'{{{RDF_NS}}}nodeID')
                if iri:
                    concept = iri
                    errors.setdefault(msg_text, []).append(concept)
                else:
                    # Treat blank nodes or missing IRI as "other" with raw payload
                    info_xml = ET.tostring(result, encoding='unicode')
                    errors["other"].setdefault(msg_text, []).append(info_xml)

        return errors

    except ET.ParseError:
        errors["Parsing"] = "Error parsing XML response:\n" + text
        return errors


def _parse_jsonld_obj(obj: dict[str, Any], errors: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a JSON-LD SHACL validation report.
    Expects a top-level dict with '@graph' and '@context', or a compact form.
    """
    # Locate the graph (standard JSON-LD)
    graph = obj.get("@graph") or obj.get("graph") or []
    if not isinstance(graph, list):
        # Some emitters may use a single node
        graph = [graph]

    # 1) Check conforms on the ValidationReport node
    report_nodes = [n for n in graph if _has_type(n, "sh:ValidationReport")]
    conforms_value = None
    if report_nodes:
        # sh:conforms can be a dict with "@value", or a plain string/boolean
        conforms = report_nodes[0].get("sh:conforms")
        if isinstance(conforms, dict):
            conforms_value = str(conforms.get("@value", "")).strip().lower()
        elif conforms is not None:
            conforms_value = str(conforms).strip().lower()

    if conforms_value == "true":
        errors["Success"] = ["Validation successful: No errors in the taxonomy"]
        return errors

    # 2) Collect all ValidationResult nodes
    result_nodes = [n for n in graph if _has_type(n, "sh:ValidationResult")]

    for res in result_nodes:
        # Message can be string or an object with '@value'
        message = res.get("sh:resultMessage")
        if isinstance(message, dict):
            msg_text = str(message.get("@value", "")).strip()
        else:
            msg_text = str(message or "").strip()

        # Focus node can be a dict with '@id' or a plain string (IRI), or missing
        focus = res.get("sh:focusNode")
        concept = None
        if isinstance(focus, dict):
            concept = focus.get("@id")
        elif isinstance(focus, str):
            concept = focus

        # Decide grouping: "message" → [concept] when IRI present and not a blank node; else put raw JSON in "other"
        raw_json = json.dumps(res, ensure_ascii=False)
        if concept and not str(concept).startswith("_:"):
            errors.setdefault(msg_text, []).append(concept)
        else:
            errors["other"].setdefault(msg_text or "Unknown message", []).append(raw_json)

    return errors


def _has_type(node: dict[str, Any], type_iri: str) -> bool:
    """
    Returns True if node's @type contains the given CURIE IRI (e.g., 'sh:ValidationReport').
    Handles @type being a string or list.
    """
    t = node.get("@type")
    if isinstance(t, list):
        return type_iri in t
    return t == type_iri

def extract_subgraph_for_uris(turtle_data: str, target_uris: list[URIRef]) -> str:
    g = Graph()
    g.parse(data=turtle_data, format='turtle')

    subgraph = Graph()
    for uri in target_uris:
        for triple in g.triples((uri, None, None)):
            subgraph.add(triple)
        for triple in g.triples((None, None, uri)):
            subgraph.add(triple)
        for triple in g.triples((None, uri, None)):
            subgraph.add(triple)

    return subgraph.serialize(format='turtle')

async def generate_explanations(model: dict[str, Any], error_msg: str, error_details: dict[str, Any], concepts: list[str], ctx: Context):
    # For each concept, call the LLM to explain the errors
    # Use model["ttl"] (JSON-LD, list of dicts) to find concept info
    list_uri = []

    if "ttl_raw" in model:
        for concept in concepts:
            if concept != "other":
                list_uri.append(URIRef(concept))
            else:
                print("N/A")
    
        sub_graph = extract_subgraph_for_uris(model["ttl_raw"], list_uri)
        if sub_graph == "\n":
            sub_graph = model["ttl_raw"]

    prompt = {
        "instruction": (
            "You are a semantic interoperability and ontology expert. "
            "Your task is to analyze a SHACL validation error for a user's data model, using the provided context. "
            "Inputs:\n"
            "- error: The SHACL validation error message, describing the issue detected.\n"
            "- SEMIC rule description: A plain-language summary of the SEMIC style guide rule that was violated.\n"
            "- SEMIC rule validation detail: How the rule is checked in practice.\n"
            "- concepts: The list of concept URIs (or a list of the complete shacl error nodes when no concept were referenced) involved in the error.\n"
            "- sub-graph: The RDF triples (in Turtle format) directly relevant to the error and the concepts involved (or the full graph when no concepts were referenced in the shacl error nodes).\n"
            "\n"
            "Instructions:\n"
            "1. Clearly identify the error and the SEMIC rule it relates to.\n"
            "2. Explain, in the context of the user's data model (using the sub-graph), what this error means and which concepts are affected.\n"
            "3. list the concerned concepts (URIs or labels) that are directly involved in the error.\n"
            "4. Provide a brief, actionable recommendation for how the user can resolve or overcome this error, referencing the SEMIC rule and the sub-graph.\n"
            "\n"
            "Output format:\n"
            "{\n"
            "  'error': <the error message>,\n"
            "  'explanation': <what this error means in the context of the user's data model>,\n"
            "  'concerned_concepts': [<list of URIs or labels>],\n"
            "  'resolution': <brief recommendation to fix the error>\n"
            "}"
        ),
        "error": error_msg,
        "SEMIC rule description": error_details["Description"],
        "SEMIC rule validation detail": error_details["How is it checked?"],
        "sub-graph": sub_graph,
    }
    response = await ctx.sample(
        messages=[json.dumps(prompt, ensure_ascii=False, indent=2)],
        system_prompt="You are a semantic interoperability and ontology expert.",
        temperature=0.0,
        max_tokens=600,
    )
    m = re.search(r"\{.*\}", getattr(response, "text", str(response)), re.S)

    return m, response

def shacl_validation(turtle_data: str, validation_server: str, output_format: str, validation_version: str) -> None:
    """  
    Validates RDF data in Turtle format using a SHACL API.  
  
    This function sends a POST request to a SHACL validation API with the RDF data. It checks the API response to determine if the RDF conforms to the SHACL shapes and prints validation results.  
  
    Parameters:  
    -----------  
    turtle_data : str  
        The RDF data in Turtle format to be validated. 
    validation_server : str  
        The API endpoint used for validating the resulting RDF file. 
    output_format : str  
        The format of the resulting rdf to be validated.
  
    Returns:  
    --------  
    None  
  
    Side Effects:  
    -------------  
    - Sends an HTTP POST request to a SHACL API.  
    - Prints validation results to the console, indicating success or failure and any errors detected.  
  
    Raises:  
    -------  
    Exception  
        If the HTTP request fails or the API returns an error response.  
    """  
    # Prepare the API request payload  
    payload = {
        "contentToValidate": turtle_data,
        "contentSyntax": output_format,
        "validationType": validation_version
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(
                validation_server,
                json=payload,
                timeout=120,
                stream=True
            )
            if response.status_code == 200:
                # Read streamed content safely
                content = b"".join(response.iter_content(chunk_size=8192))
                text = content.decode("utf-8", errors="replace")
                return parse_validator_output(text)
            else:
                return {"Error": f"Error with the API call to ITB validator: {response.status_code} {response.text}"}

        except requests.exceptions.ChunkedEncodingError as e:
            print(f"[WARN] ChunkedEncodingError on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                return {"Error": f"ChunkedEncodingError after {retries} attempts: {str(e)}"}

        except requests.exceptions.Timeout:
            return {"Error": "Request to ITB validator timed out."}
            
async def validator_check(
    user: str,
    name: str,
    ctx: Context = None,
    validation_server: str = None,
    output_format: str = "text/turtle",
    validation_version: str = "owl",
) -> dict:
    """Tool for validating a data model against the SEMIC style guide conventions using SHACL and generating human-readable explanations for each problematic concept using an LLM."""
    model = get_model(user, name)
    # Get the Turtle data from the model (assume model["ttl_data"] or similar)
    # You may need to adapt this depending on your model structure
    if "ttl_raw" in model.keys():
        model_data = model["ttl_raw"]
    else:
        model_data = model

    # Call SHACL validation
    errors = shacl_validation(
        model_data,
        validation_server,
        output_format,
        validation_version
    )

    explanations = {}
    i = 0
    for error_msg, concepts in errors.items():
        i = i + 1

        await ctx.report_progress(progress=i, total=len(errors.keys()))

        if i % 30 == 0 and i > 0:
            await ctx.close_sse_stream()

        print("ERROR MSG:", error_msg)
        error_prefix = error_msg.split(":", 1)[0]
        if error_prefix in RULES_DICT.keys():
            error_details = RULES_DICT[error_prefix]
            m, response = await generate_explanations(model, error_msg, error_details, concepts, ctx)
            try:
                explanations[error_msg] = json.loads(m.group(0) if m else response.text)
            except Exception:
                explanations[error_msg] = {"llm_output": getattr(response, "text", str(response))}

        elif error_msg == "other":
            explanations["other"] = {}
            print("Other")
        
        elif error_msg == "Success":
            explanations[error_msg] = {"message": "Validation successful: No errors in the taxonomy"}
        
        else: 
            explanations[error_msg] = {"llm_output": "No SEMIC rule found for this error."}
    
    return explanations


validator_check.__doc__ = f"""
    Tool for validating a data model against the SEMIC style guide using SHACL and generating LLM-based explanations for each problematic concept.

    This tool:
      - Loads the model and extracts its Turtle serialization.
      - Calls a SHACL validation API to identify conformance errors.
      - For each concept with errors, calls an LLM to generate a human-readable explanation and suggestions for resolution.

    Args:
        user (str): Username or model owner.
        name (str): Model file name (without .json extension).
        ctx (Context): LLM context object.
        validation_server (str): SHACL validation API endpoint. Limited to one of the two following values: "https://www.itb.ec.europa.eu/shacl/semicstyleguide/api/validate" if the user's data model is in ttl format or "https://www.itb.ec.europa.eu/shacl/model2owl/api/validate" if it's in XMI format.
        output_format (str): RDF format (e.g., "text/turtle").
        validation_version (str): Depending on the format of the user's data model. Limited to one of the three following values: "owl", "shacl", "uml". "owl" should be used if the data model is an owl ontology, "shacl" if the data model is composed of shacl shapes, and "uml" if the data model is in XMI format.

    Returns:
        dict: Mapping from concept URI (or "other") to a dictionary with a human-readable explanation of the SHACL validation errors.

    Notes:
        - Only SHACL-detectable issues are reported; further style guide checks may be needed.
        - LLM explanations are per concept and help users understand and resolve the issues.
"""
