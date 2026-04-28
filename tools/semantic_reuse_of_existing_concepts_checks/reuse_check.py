from __future__ import annotations
from resources.semantic_model.utils import get_model
from tools.index_search import retrieve_documents
from fastmcp import Context

# --- JSON-LD helpers ---
def _get_ld_label(item):
    # rdfs:label
    labels = item.get("http://www.w3.org/2000/01/rdf-schema#label", [])
    for label in labels:
        if label.get("@language") == "en":
            return label.get("@value")
    if labels:
        return labels[0].get("@value")
    return None

async def reuse_check(
    user: str,
    name: str,
    ctx: Context = None,
    vocabularies: list = None,
    n_documents: int = 5,
    target_names: list = None,
) -> dict:
    """Tool for assessing semantic interoperability of a data model by checking reuse of standard concepts."""
    import json
    model = get_model(user, name)
    results = {}

    # Concrete guidance from the style guide
    reuse_guidance = (
        "Proper reuse means: "
        "- If reusing a class as-is, adopt the original URI, label, and definition with NO changes. "
        "- If terminological adaptations are needed, create a subclass and clearly indicate the adaptation, but do not change the semantics. "
        "- If semantic adaptations are needed, create a subclass with a new label and definition, and document the reuse chain. "
        "- When reusing, all mandatory properties from the original should be included, and optional ones only if relevant. "
        "- Do not duplicate properties from superclasses, and avoid logical contradictions. "
        "- Always make reuse explicit with notes, hyperlinks, or dereferenceable URIs. "
        "- If no relevant standard is found, suggest domain standards the user could look at, but do not provide further recommendations."
    )

    # Detect model type
    if "elements" in model and "connectors" in model:
        # UML XMI JSON
        if target_names:
            class_names = set(target_names)
        else:
            class_names = {element.get("name") for element in model.get("elements", []) if element.get("type") == "uml:Class"}

        for i, element in enumerate(model.get("elements", [])):
            
            await ctx.report_progress(progress=i, total=len(model.get("elements", [])))

            if i % 30 == 0 and i > 0:
                await ctx.close_sse_stream()

            if element.get("type") == "uml:Class" and element.get("name") in class_names:
                class_name = element.get("name")
                # Collect class and its attributes
                class_with_properties = dict(element)
                class_with_properties["attributes"] = element.get("attributes", [])
                docs = retrieve_documents(class_name, vocabularies or [], n_documents)
                prompt = {
                    "instruction": (
                        "You are a semantic interoperability expert. "
                        "Given the user's class definition (including its attributes/properties) and the retrieved standard class documentation, "
                        "analyze the reuse of standards for this class according to the following criteria: "
                        f"{reuse_guidance} "
                        "Return a JSON dictionary with the following keys: "
                        "'relevant_standard': the most relevant standard class to reuse (if any in the candidate, else null or empty), "
                        "'general_comment': a short comment on whether there is proper reuse and, if not, why based on the list of criteria, "
                        "'recommendations': concrete recommendations for improving reuse or interoperability. "
                        "If no relevant standards are found, set 'relevant_standard' to null, and provide suggestions in 'recommendations'."
                    ),
                    "user_class": class_with_properties,
                    "candidate_standards": docs
                }
                response = await ctx.sample(
                    messages=[json.dumps(prompt, ensure_ascii=False, indent=2)],
                    system_prompt="You are a semantic interoperability expert.",
                    temperature=0.0,
                    max_tokens=800,
                )
                import json, re
                m = re.search(r"\{.*\}", getattr(response, "text", str(response)), re.S)
                try:
                    results[class_name] = json.loads(m.group(0) if m else response.text)
                except Exception:
                    results[class_name] = {"llm_output": getattr(response, "text", str(response))}
                print("[DEBUG] Processed class:")
                print(results[class_name])
            

    elif "ttl" in model:
        # JSON-LD
        # Collect all OWL classes and SHACL NodeShapes and their properties
        class_map = {}
        node_shape_map = {}
        for item in model["ttl"]:
            item_type = item.get("@type", [])
            # OWL Class
            if "http://www.w3.org/2002/07/owl#Class" in item_type:
                label = _get_ld_label(item)
                if target_names:
                    if label and label in target_names:
                        class_map[item.get("@id")] = item
                else:
                    class_map[item.get("@id")] = item
            # SHACL NodeShape
            if (
                (isinstance(item_type, str) and item_type == "sh:NodeShape") or
                (isinstance(item_type, list) and "sh:NodeShape" in item_type)
            ):
                # Use sh:name as label if available, else @id
                sh_name = item.get("sh:name", {})
                label = sh_name.get("en") if isinstance(sh_name, dict) else None
                if not label:
                    label = item.get("@id")
                if target_names:
                    if label and label in target_names:
                        node_shape_map[item.get("@id")] = item
                else:
                    node_shape_map[item.get("@id")] = item
        i = 0
        # Process OWL classes
        for class_id, class_obj in class_map.items():
            i = i + 1

            await ctx.report_progress(progress=i, total=len(class_map.keys()))

            if i % 30 == 0 and i > 0:
                await ctx.close_sse_stream()
            
            label = _get_ld_label(class_obj)
            # Find all properties with this class as domain
            properties = []
            for item in model["ttl"]:
                item_type = item.get("@type", [])
                if ("http://www.w3.org/2002/07/owl#ObjectProperty" in item_type or
                    "http://www.w3.org/2002/07/owl#DatatypeProperty" in item_type):
                    for domain in item.get("http://www.w3.org/2000/01/rdf-schema#domain", []):
                        if domain.get("@id") == class_id:
                            properties.append(item)
            class_with_properties = dict(class_obj)
            class_with_properties["properties"] = properties
            docs = retrieve_documents(label, vocabularies or [], n_documents)
            prompt = {
                "instruction": (
                    "You are a semantic interoperability expert. "
                    "Given the user's class definition (including its properties) and the retrieved standard class documentation, "
                    "analyze the reuse of standards for this class according to the following criteria: "
                    f"{reuse_guidance} "
                    "Return a JSON dictionary with the following keys: "
                    "'relevant_standard': the most relevant standard class to reuse (if any in the candidate, else null or empty), "
                    "'general_comment': a short comment on whether there is proper reuse and, if not, why based on the list of criteria, "
                    "'recommendations': concrete recommendations for improving reuse or interoperability. "
                    "If no relevant standards are found, set 'relevant_standard' to null, and provide suggestions in 'recommendations'."
                ),
                "user_class": class_with_properties,
                "candidate_standards": docs
            }
            response = await ctx.sample(
                messages=[json.dumps(prompt, ensure_ascii=False, indent=2)],
                system_prompt="You are a semantic interoperability expert.",
                temperature=0.0,
                max_tokens=800,
            )
            import json, re
            m = re.search(r"\{.*\}", getattr(response, "text", str(response)), re.S)
            try:
                results[label] = json.loads(m.group(0) if m else response.text)
            except Exception:
                results[label] = {"llm_output": getattr(response, "text", str(response))}
            print("[DEBUG] Processed class:")
            print(results[label])

    else:
        raise ValueError("Unknown model format: expected UML XMI or JSON-LD with 'ttl' key")

    return {"Compliance with SEMIC rule GC-R1: Reuse existing concepts as much as possible, respecting the original semantics and lexicalisation.": results}


reuse_check.__doc__ = f"""
    Tool for checking the reuse of standard concepts in a data model.

    Modes:
      1. Full model check: If target_names is None, checks all classes in the model.
      2. Targeted check: If target_names is provided (list of class names), only those classes are checked.

    For each selected class:
      - Searches the index for relevant standard classes from vocabularies.
      - If relevant standards are found, checks if the model is properly reusing them according to SEMIC style guide:
          * Reuse as-is: adopt original URI, label, and definition with NO changes.
          * Terminological adaptation: create a subclass, indicate adaptation, do not change semantics.
          * Semantic adaptation: create a subclass with new label/definition, document reuse chain.
          * Include all mandatory properties from the original, optional ones only if relevant.
          * Do not duplicate properties from superclasses, avoid logical contradictions.
          * Make reuse explicit with notes, hyperlinks, or dereferenceable URIs.
      - If no relevant standards are found, suggests domain standards the user could look at, but does not provide further recommendations.

    Args:
        vocabularies (list, optional):
            list of vocabularies to restrict the search. Defaults to None (all vocabularies).
        n_documents (int, optional):
            Number of documents to retrieve from the index. Defaults to 5.
        target_names (list, optional):
            list of class names to check. If None, all classes are checked.

    Returns:
        dict:
            Mapping from class name to a dictionary with keys:
                - 'relevant_standard': the most relevant standard class to reuse (if any),
                - 'general_comment': a short comment on whether there is proper reuse and, if not, why,
                - 'recommendations': concrete recommendations for improving reuse or interoperability.
            If no relevant standards are found, 'relevant_standard' is null and 'recommendations' contains suggestions.
"""
