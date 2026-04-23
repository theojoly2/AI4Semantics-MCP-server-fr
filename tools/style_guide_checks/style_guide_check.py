from __future__ import annotations
import json
from fastmcp import Context
from .conventions import metadata_conventions, reuse_conventions


async def style_guide_check(
    ctx: Context = None,
    validator_check: dict = None,
    metadata_checks: dict = None,
    reuse_checks: dict = None,
) -> dict:

    """Generates a structured semantic interoperability assessment report for a data model using LLM summarization."""
    print("Validator Check:")
    print(json.dumps(validator_check, indent=2, ensure_ascii=False))

    if validator_check is None:
        validator_check = {}
    if metadata_checks is None:
        metadata_checks = {}
    if reuse_checks is None:
        reuse_checks = {}

    # Fixed report structure
    report_template = (
        "# Semantic Style Guide Assessment Report\n\n"
        "{{validation_section}}\n\n"
        "{{metadata_section}}\n\n"
        "{{reuse_section}}\n"
    )

    # Compose the validator section prompt
    validator_prompt = (
        "You are a semantic data modeling expert. Given the following checks of a data model against the SEMIC style guide validator, "
        "write a summary for the section 'Validation against the ITB validator' as follows: For each rule that was not respected, create a subsection with the rule name or identifier as the heading. In each subsection, provide: (1) a summary explaining the error noted, (2) if available, an example from the data model illustrating the error, (3) a clear suggestion for how to solve the error, and (4) a list of problematic concepts or elements. Use only the information found in validator_checks. Structure the output in markdown.\n\n"
        f"Validator checks input:\n{json.dumps(validator_check, indent=2, ensure_ascii=False)}"
    )
    
    # Compose the metadata section prompt
    metadata_prompt = (
        f"You are a semantic data modeling expert. Given the following checks of a data model against a set of SEMIC style guide conventions: {metadata_conventions} \n"
        "Write a summary for the metadata quality section as follows: For each convention that was not respected, create a subsection with the convention name as the heading. In each subsection, provide: (1) a summary explaining the error noted, (2) if available, an example from the data model illustrating the error, (3) a clear suggestion for how to solve the error, and (4) a list of problematic concepts or elements. Use only the information found in metadata_checks. Structure the output in markdown.\n\n"
        f"Metadata checks input:\n{json.dumps(metadata_checks, indent=2, ensure_ascii=False)}"
    )

    # Compose the reuse section prompt
    reuse_prompt = (
        f"You are a semantic interoperability expert. Given the following reuse checks for each class against a set of SEMIC style guide conventions: {reuse_conventions} \n"
        "Write a summary for the reuse of standards section as follows: For each convention that was not respected, create a subsection with the convention name as the heading. In each subsection, provide: (1) a summary explaining the error noted, (2) if available, an example from the data model illustrating the error, (3) a clear suggestion for how to solve the error, and (4) a list of problematic concepts or elements. Use only the information found in reuse_checks. Structure the output in markdown.\n\n"
        f"Reuse checks input:\n{json.dumps(reuse_checks, indent=2, ensure_ascii=False)}"
    )

    # Get LLM summaries for each section
    if validator_check:
        validator_response = await ctx.sample(
            messages=[validator_prompt],
            system_prompt="You are a semantic data modeling expert writing a standards-based assessment report.",
            temperature=0.0,
            max_tokens=3000,
        )
    else: 
        validator_response = ""
    
    if metadata_checks:
        metadata_response = await ctx.sample(
            messages=[metadata_prompt],
            system_prompt="You are a semantic data modeling expert writing a standards-based assessment report.",
            temperature=0.0,
            max_tokens=3000,
        )
    else:
        metadata_response = ""

    if reuse_checks:
        reuse_response = await ctx.sample(
            messages=[reuse_prompt],
            system_prompt="You are a semantic interoperability expert writing a standards-based assessment report.",
            temperature=0.0,
            max_tokens=3000,
        )
    else:
        reuse_response = ""

    # Extract text from LLM responses
    validation_section = getattr(validator_response, "text", str(validator_response)).strip()
    metadata_section = getattr(metadata_response, "text", str(metadata_response)).strip()
    reuse_section = getattr(reuse_response, "text", str(reuse_response)).strip()

    # Fill the template
    report = report_template.replace("{{validation_section}}", validation_section).replace("{{metadata_section}}", metadata_section).replace("{{reuse_section}}", reuse_section)

    return {"report": report}


style_guide_check.__doc__ = f"""
    This tool produces a markdown report with a fixed structure, summarizing the results of an assessment of data model against the SEMIC style guide convention.
    Args:
        ctx (Context, optional):
            The LLM context for prompt completion.

    Returns:
        dict:
            A dictionary with a single key 'report' containing the full markdown assessment report as a string.

    Notes:
        - The report is intended to provide a standards-based, human-readable assessment of a data model's semantic interoperability, metadata quality, and reuse of standards.
        - All sections are generated using LLM summarization for clarity and actionable recommendations.
"""
