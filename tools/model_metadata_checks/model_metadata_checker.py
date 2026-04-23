from __future__ import annotations
from resources.semantic_model.utils import get_model
from tools.style_guide_validator import extract_subgraph_for_uris
from fastmcp import Context
from rdflib import URIRef
from config import load_config
from pathlib import Path
from typing import Any
import pandas as pd
import json
import re

config = load_config()

STYLE_GYDE_XLS_PATH = Path(config["file_paths"]["style_guide_xls"])

df = pd.read_excel(STYLE_GYDE_XLS_PATH, index_col="Rule")
RULES_DICT = df.to_dict(orient="index")

def extract_classes_associations(model: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """
    Extracts classes and associations from the model whose names are in the provided list.
    Supports both UML XMI JSON and JSON-LD (ttl) formats.
    """
    result = {
        "classes": [],
        "associations": []
    }

    # UML XMI JSON
    if "elements" in model and "connectors" in model:
        for element in model.get("elements", []):
            if element.get("type") == "uml:Class" and element.get("name") in names:
                result["classes"].append(element)
        for connector in model.get("connectors", []):
            if (
                connector.get("relationship") == "Association" and
                (connector.get("name") in names or
                 connector.get("source_name") in names or
                 connector.get("target_name") in names)
            ):
                result["associations"].append(connector)
    # JSON-LD (ttl key)
    elif "ttl" in model:
        for item in model["ttl"]:
            item_type = item.get("@type", [])
            if "http://www.w3.org/2002/07/owl#Class" in item_type:
                label = _get_ld_label(item)
                if label and label in names:
                    result["classes"].append(item)
            if ("http://www.w3.org/2002/07/owl#ObjectProperty" in item_type or
                "http://www.w3.org/2002/07/owl#DatatypeProperty" in item_type):
                label = _get_ld_label(item)
                if label and label in names:
                    result["associations"].append(item)
    else:
        raise ValueError("Unknown model format: expected UML XMI or JSON-LD with 'ttl' key")

    return result


# ---------------------------------------------------------------------------
# Helpers to read tag values from XMI elements / connectors
# ---------------------------------------------------------------------------

def _xmi_tag(tags: list[dict], tag_name: str) -> str | None:
    """Return the value of the first tag matching tag_name, or None."""
    for t in tags:
        if t.get("name") == tag_name:
            return t.get("value") or None
    return None


def _xmi_concept_subgraph(concept: dict, concept_type: str) -> str:
    """
    Build a compact human-readable description of an XMI class or connector
    to use as the 'sub-graph' sent to the LLM for R4/R5/R7 checks.
    Mirrors the Turtle sub-graph used in the TTL branch.
    """
    lines = [f"# {concept_type}: {concept.get('name', '(unnamed)')}"]
    tags = concept.get("tags", [])

    uri     = _xmi_tag(tags, "uri")
    label   = _xmi_tag(tags, "label-en")
    defn    = _xmi_tag(tags, "definition-en")
    usage   = _xmi_tag(tags, "usageNote-en")

    if uri:
        lines.append(f"  uri: {uri}")
    if label:
        lines.append(f"  label (en): {label}")
    if defn:
        lines.append(f"  definition (en): {defn}")
    if usage:
        lines.append(f"  usage note (en): {usage}")

    # Include attributes for classes
    for attr in concept.get("attributes", []):
        attr_tags = attr.get("tags_attribute", [])
        a_uri   = _xmi_tag(attr_tags, "uri")
        a_label = _xmi_tag(attr_tags, "label-en")
        a_def   = _xmi_tag(attr_tags, "definition-en")
        a_usage = _xmi_tag(attr_tags, "usageNote-en")
        lines.append(f"  # attribute: {attr.get('name', '(unnamed)')} [{attr.get('type', '')}]")
        if a_uri:
            lines.append(f"    uri: {a_uri}")
        if a_label:
            lines.append(f"    label (en): {a_label}")
        if a_def:
            lines.append(f"    definition (en): {a_def}")
        if a_usage:
            lines.append(f"    usage note (en): {a_usage}")

    # Include target-end tags for connectors
    for t in concept.get("tags_target", []):
        lines.append(f"  target tag [{t.get('name')}]: {t.get('value')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GC-R3: metadata completeness
# ---------------------------------------------------------------------------

def metadata_checks(model: dict) -> dict[str, list[str]]:
    """
    Check if all classes, attributes, and associations have the required metadata:
    URI, definition-en, label-en, usage_note_en.

    Returns a dict mapping concept identifiers to a list of missing metadata items.
    Supports both UML XMI JSON and JSON-LD (ttl) formats.
    """
    problematic_concepts: dict[str, list[str]] = {}

    def _flag(concept_id: str, issue: str):
        problematic_concepts.setdefault(concept_id, []).append(issue)

    # ------------------------------------------------------------------
    # UML XMI JSON
    # ------------------------------------------------------------------
    if "elements" in model and "connectors" in model:
        for element in model["elements"]:
            if element.get("type") != "uml:Class":
                continue

            name = element.get("name", "(unnamed class)")
            tags = element.get("tags", [])
            tag_names = {t.get("name") for t in tags}

            if "uri" not in tag_names:
                _flag(f"Class: {name}", "missing URI")
            if "definition-en" not in tag_names:
                _flag(f"Class: {name}", "missing definition")
            if "label-en" not in tag_names:
                _flag(f"Class: {name}", "missing label")
            if "usageNote-en" not in tag_names:
                _flag(f"Class: {name}", "missing usage note")

            for attribute in element.get("attributes", []):
                attr_name = attribute.get("name", "(unnamed attribute)")
                concept_id = f"Attribute: {name}/{attr_name}"
                attr_tag_names = {t.get("name") for t in attribute.get("tags_attribute", [])}

                if "uri" not in attr_tag_names:
                    _flag(concept_id, "missing URI")
                if "definition-en" not in attr_tag_names:
                    _flag(concept_id, "missing definition")
                if "label-en" not in attr_tag_names:
                    _flag(concept_id, "missing label")
                if "usageNote-en" not in attr_tag_names:
                    _flag(concept_id, "missing usage note")

        for connector in model["connectors"]:
            if connector.get("relationship") != "Association":
                continue

            rt = connector.get("rt") or connector.get("source_name", "(unnamed)")
            concept_id = f"Association: {connector.get('source_name', '')}/{rt}"
            tgt_tag_names = {t.get("name") for t in connector.get("tags_target", [])}

            if "uri" not in tgt_tag_names:
                _flag(concept_id, "missing URI")
            if "definition-en" not in tgt_tag_names:
                _flag(concept_id, "missing definition")
            if "label-en" not in tgt_tag_names:
                _flag(concept_id, "missing label")
            if "usageNote-en" not in tgt_tag_names:
                _flag(concept_id, "missing usage note")

    # ------------------------------------------------------------------
    # JSON-LD (ttl key)
    # ------------------------------------------------------------------
    elif "ttl" in model:
        for item in model["ttl"]:
            item_id   = item.get("@id", "")
            item_type = item.get("@type", [])

            is_class = "http://www.w3.org/2002/07/owl#Class" in item_type
            is_prop  = (
                "http://www.w3.org/2002/07/owl#ObjectProperty" in item_type or
                "http://www.w3.org/2002/07/owl#DatatypeProperty" in item_type
            )
            if not (is_class or is_prop):
                continue

            if not item_id:
                _flag("[no id]", "missing URI")

            if not _get_ld_label(item):
                _flag(item_id, "missing label")
            if not _get_ld_comment(item):
                _flag(item_id, "missing definition")
            if not _get_ld_usage_note(item):
                _flag(item_id, "missing usage note")

    else:
        raise ValueError("Unknown model format: expected UML XMI or JSON-LD with 'ttl' key")

    return problematic_concepts


# ---------------------------------------------------------------------------
# JSON-LD helpers
# ---------------------------------------------------------------------------

def _get_ld_label(item):
    labels = item.get("http://www.w3.org/2000/01/rdf-schema#label", [])
    for label in labels:
        if label.get("@language") == "en":
            return label.get("@value")
    if labels:
        return labels[0].get("@value")
    return None

def _get_ld_comment(item):
    comments = item.get("http://www.w3.org/2000/01/rdf-schema#comment", [])
    for comment in comments:
        if comment.get("@language") == "en":
            return comment.get("@value")
    if comments:
        return comments[0].get("@value")
    return None

def _get_ld_usage_note(item):
    notes = item.get("http://www.w3.org/2004/02/skos/core#scopeNote", [])
    for note in notes:
        if note.get("@language") == "en":
            return note.get("@value")
    if notes:
        return notes[0].get("@value")
    return None


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

async def generate_explanations(rule_description: str, uri: str, sub_graph: str, ctx: Context):
    prompt = {
        "instruction": (
            "You are a semantic interoperability and ontology expert. "
            "Your task is to analyze a specific concept from a user's data model and check whether it violates a specific set of design rules, using the provided context. "
            "Inputs:\n"
            "- SEMIC rule description: A plain-language summary of the SEMIC style guide rule that needs to be checked.\n"
            "- concept: The concept URI (or label) that needs to be analysed against the rules.\n"
            "- sub-graph: The metadata directly relevant to the concept involved.\n"
            "\n"
            "Instructions:\n"
            "1. Clearly identify the potential error and the SEMIC rule it relates to.\n"
            "2. Explain, in the context of the user's data model (using the sub-graph), what this error means and which concepts are affected.\n"
            "3. List the concerned concepts (URIs or labels) that are directly involved in the error.\n"
            "4. Provide a brief, actionable recommendation for how the user can resolve or overcome this error, referencing the SEMIC rule and the sub-graph.\n"
            "\n"
            "Output format:\n"
            "{\n"
            "  'error': <the error message>,\n"
            "  'explanation': <what this error means in the context of the user's data model>,\n"
            "  'concerned_concept': URI or label,\n"
            "  'resolution': <brief recommendation to fix the error>\n"
            "}"
        ),
        "SEMIC rule description": rule_description,
        "concept": uri,
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


# ---------------------------------------------------------------------------
# GC-R4 / R5 / R7: terminology & definition consistency (LLM-assisted)
# ---------------------------------------------------------------------------

async def R4_5_7_checks(model: dict[str, Any], ctx: Context):

    non_observance_of_GC_R4 = {
        "error": "Non-observance of SEMIC rule GC-R4: The terminology style shall be consistent across the vocabulary.",
        "explanation": RULES_DICT["Non-observance of SEMIC rule GC-R4"]["Description"],
        "concerned_concepts": {},
        "resolution": "See details per concept",
    }
    non_observance_of_GC_R5 = {
        "error": "Non-observance of SEMIC rule GC-R5: The concept definitions shall be elaborated consistently across the vocabulary.",
        "explanation": RULES_DICT["Non-observance of SEMIC rule GC-R5"]["Description"],
        "concerned_concepts": {},
        "resolution": "See details per concept",
    }
    non_observance_of_GC_R7 = {
        "error": "Non-observance of SEMIC rule GC-R7: Indicators of deontic modalities for classes and properties do not have semantic or normative value. Still they may be used as editorial annotations.",
        "explanation": RULES_DICT["Non-observance of SEMIC rule GC-R7"]["Description"],
        "concerned_concepts": {},
        "resolution": "See details per concept",
    }

    async def _check_concept(concept_id: str, sub_graph: str):
        """Run all three LLM rule checks for a single concept and store results."""
        for rule_key, bucket in (
            ("Non-observance of SEMIC rule GC-R4", non_observance_of_GC_R4),
            ("Non-observance of SEMIC rule GC-R5", non_observance_of_GC_R5),
            ("Non-observance of SEMIC rule GC-R7", non_observance_of_GC_R7),
        ):
            m, response = await generate_explanations(
                RULES_DICT[rule_key]["Description"], concept_id, sub_graph, ctx
            )
            try:
                bucket["concerned_concepts"][concept_id] = json.loads(
                    m.group(0) if m else response.text
                )
            except Exception:
                bucket["concerned_concepts"][concept_id] = {
                    "llm_output": getattr(response, "text", str(response))
                }

    # ------------------------------------------------------------------
    # UML XMI JSON
    # ------------------------------------------------------------------
    if "elements" in model and "connectors" in model:
        # Collect all concepts: classes (+ their attributes inline) and associations.
        concepts: list[tuple[str, str]] = []   # (concept_id, sub_graph_text)

        for element in model.get("elements", []):
            if element.get("type") != "uml:Class":
                continue
            tags   = element.get("tags", [])
            uri    = _xmi_tag(tags, "uri") or element.get("name", "(unnamed)")
            sub_graph = _xmi_concept_subgraph(element, "Class")
            concepts.append((uri, sub_graph))

        for connector in model.get("connectors", []):
            if connector.get("relationship") != "Association":
                continue
            tgt_tags = connector.get("tags_target", [])
            uri = (
                _xmi_tag(tgt_tags, "uri")
                or connector.get("rt")
                or f"{connector.get('source_name', '')} -> {connector.get('target_name', '')}"
            )
            sub_graph = _xmi_concept_subgraph(connector, "Association")
            concepts.append((uri, sub_graph))

        for i, (concept_id, sub_graph) in enumerate(concepts):
            await ctx.report_progress(progress=i, total=len(concepts))
            # Throttle: close/reopen SSE stream every 30 items to avoid timeouts.
            if i % 30 == 0 and i > 0:
                await ctx.close_sse_stream()
            await _check_concept(concept_id, sub_graph)

    # ------------------------------------------------------------------
    # JSON-LD (ttl key)
    # ------------------------------------------------------------------
    elif "ttl" in model:
        for i, item in enumerate(model["ttl"]):
            await ctx.report_progress(progress=i, total=len(model["ttl"]))
            if i % 30 == 0 and i > 0:
                await ctx.close_sse_stream()

            item_id   = item.get("@id", "")
            item_type = item.get("@type", [])

            if not (
                "http://www.w3.org/2002/07/owl#Class" in item_type or
                "http://www.w3.org/2002/07/owl#ObjectProperty" in item_type or
                "http://www.w3.org/2002/07/owl#DatatypeProperty" in item_type
            ):
                continue

            sub_graph = extract_subgraph_for_uris(model["ttl_raw"], [URIRef(item_id)])
            if sub_graph == "\n":
                sub_graph = model["ttl_raw"]

            await _check_concept(item_id, sub_graph)

    else:
        msg = {"error": "Unknown model format: expected UML XMI JSON or OWL ontology in Turtle"}
        return msg, msg, msg

    return non_observance_of_GC_R4, non_observance_of_GC_R5, non_observance_of_GC_R7


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def metadata_checker(
    user: str = "",
    name: str = "",
    target_names: list[str] = None,
    check_instruction: str = None,
    ctx: Context = None,
) -> dict:
    """
    Validate metadata completeness and terminology consistency in a semantic model
    based on general conventions of the SEMIC style guide.

    This tool checks a model (loaded via `get_model(user, name)`) for:
      • GC‑R3 — Metadata completeness (URI, definition, label, usage note)
      • GC‑R4 — Consistent terminology style
      • GC‑R5 — Consistent definition elaboration
      • GC‑R7 — Avoid deontic modality indicators as semantic/normative values

    It supports two modes:

    1) Full model check (default — when `target_names` is None):
       - For UML XMI JSON models (keys "elements" & "connectors"), it inspects classes,
         attributes, and associations by scanning their tag containers for:
         "uri", "definition-en", "label-en", and "usageNote-en".
       - For JSON‑LD models (key "ttl"), it inspects OWL Classes and Properties by reading:
         rdfs:label (en), rdfs:comment (en), skos:scopeNote (en), and @id (URI).
       - Returns a structured report for GC‑R3, GC‑R4, GC‑R5, GC‑R7.

    2) Targeted check (when `target_names` is provided):
       - Extracts only the specified classes/associations from the model.
       - Sends the extracted subset to an LLM with a custom instruction.
       - Requires `ctx`. Returns the LLM's JSON output (or raw text under "llm_output").

    Supported model formats:
      - UML XMI JSON: expects top-level keys "elements" and "connectors".
      - JSON‑LD:      expects top-level key "ttl" and "ttl_raw" (for R4/R5/R7 checks).

    Args:
        user (str):              Identifier passed to `get_model`. Defaults to "".
        name (str):              Model name passed to `get_model`. Defaults to "".
        target_names (list[str] | None): When provided, runs a targeted check.
        check_instruction (str | None):  Custom LLM instruction for targeted mode.
        ctx (Context | None):    fastmcp Context. Required for R4/R5/R7 and targeted checks.

    Returns:
        dict: Structured report with GC-R3/R4/R5/R7 sections, or LLM output for targeted checks.
    """
    model = get_model(user, name)

    if not target_names:
        non_observance_of_GC_R3 = {
            "error": "Non-observance of SEMIC rule GC-R3: All classes, attributes and associations should have a URI, a definition, a label, and ideally a usage note.",
            "explanation": RULES_DICT["Non-observance of SEMIC rule GC-R3"]["Description"],
            "concerned_concepts": metadata_checks(model),
            "resolution": "Ensure all classes, attributes, and associations have a URI, definition, label, and usage note.",
        }
        non_observance_of_GC_R4, non_observance_of_GC_R5, non_observance_of_GC_R7 = (
            await R4_5_7_checks(model, ctx)
        )

        return {
            "Non-observance of SEMIC rule GC-R3": non_observance_of_GC_R3,
            "Non-observance of SEMIC rule GC-R4": non_observance_of_GC_R4,
            "Non-observance of SEMIC rule GC-R5": non_observance_of_GC_R5,
            "Non-observance of SEMIC rule GC-R7": non_observance_of_GC_R7,
        }

    else:
        subset = extract_classes_associations(model, target_names)
        if not check_instruction:
            check_instruction = (
                "Check the following classes and associations for metadata completeness. "
                "Report any missing or incomplete metadata fields (URI, definition-en, label-en, usageNote-en)."
            )
        prompt = {"instruction": check_instruction, "data": subset}

        if ctx is None:
            raise ValueError("ctx (LLM context) must be provided for targeted checks.")

        response = await ctx.sample(
            messages=[json.dumps(prompt, ensure_ascii=False, indent=2)],
            system_prompt="You are a metadata quality checker for semantic models.",
            temperature=0.0,
            max_tokens=800,
        )
        m = re.search(r"\{.*\}", getattr(response, "text", str(response)), re.S)
        try:
            return json.loads(m.group(0) if m else response.text)
        except Exception:
            return {"llm_output": getattr(response, "text", str(response))}


metadata_checker.__doc__ = f"""
    Validate metadata completeness and terminology consistency in a semantic model based on general conventions of the SEMIC style guide.      

    This tool checks a model (loaded via `get_model(user, name)`) for:
      • GC‑R3 — Metadata completeness (URI, definition, label, usage note)
      • GC‑R4 — Consistent terminology style
      • GC‑R5 — Consistent definition elaboration
      • GC‑R7 — Avoid deontic modality indicators as semantic/normative values

    It supports two modes:

    1) Full model check (default — when `target_names` is None):
       - For UML XMI JSON models (keys: "elements" & "connectors"), it inspects classes, attributes,
         and associations by scanning their tag containers for:
           "uri", "definition-en", "label-en", and "usage_note_en".
       - For JSON‑LD models (key: "ttl"), it inspects OWL Classes and Properties by reading:
           rdfs:label (en), rdfs:comment (en), skos:scopeNote (en), and @id (URI).
       - It returns a structured report for GC‑R3, GC‑R4, GC‑R5, GC‑R7.

    2) Targeted check (when `target_names` is provided):
       - Extracts only the specified classes/associations from the model (UML XMI JSON or JSON‑LD).
       - Sends the extracted subset to an LLM with a custom instruction (`check_instruction`, or a default).
       - Requires `ctx` (LLM context). Returns the LLM's JSON output (or the raw text under "llm_output").

    Supported model formats:
      - UML XMI JSON: expects top-level keys "elements" and "connectors".
      - JSON‑LD: expects top-level key "ttl" (list of JSON‑LD nodes) and "ttl_raw" (Turtle) for R4/R5/R7 checks.

    Args:
        user (str, optional):
            Identifier passed to `get_model` to locate the model. Defaults to "".
        name (str, optional):
            Model name passed to `get_model` to locate the model. Defaults to "".
        target_names (list[str] | None, optional):
            When provided, runs a targeted check for only these class/association names.
            When None, runs the full model check.
        check_instruction (str | None, optional):
            Custom instruction for the LLM in targeted mode. If omitted, a default instruction
            to assess URI/definition/label/usage note completeness is used.
        ctx (Context | None, optional):
            fastmcp Context used to call the LLM. Required for targeted checks.
            In full model mode, ctx is used to generate explanations for GC‑R4/R5/R7.


    Returns:
        dict:
            For full model check:
                - dictionary with four SEMIC rule sections (GC‑R3, GC‑R4, GC‑R5, GC‑R7).
            Each section includes:
              - "error": short rule violation message
              - "explanation": rule description
              - "concerned_concepts": per‑concept findings
              - "resolution": suggested next step

            For targeted check:
                - LLM-generated output, typically a dict describing missing or incomplete metadata for the specified classes/associations.
"""
