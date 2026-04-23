from resources.semantic_conventions.style_guide.utils import get_style_guide as _get_style_guide


async def get_style_guide() -> str:
    return await _get_style_guide()


get_style_guide.__doc__ = """
    Returns the SEMIC style guide content.

    This function returns a set of conventions and information related to the SEMIC style guide.
    This document defines the style guide to be applied to the SEMIC's semantic data specifications,
    notably to the eGovernment Core Vocabularies and Application Profiles.

    The style guide provides rules on:
    - Naming conventions
    - Syntax
    - Artefact management and organisation

    It is meant to be complemented with technical artefacts and implementations that enable automatic
    conformance checking and transformation of conceptual models into formal semantic representations.

    The content of these guides is part of the action to promote semantic interoperability amongst
    the EU Member States, with the objective of fostering the use of standards by offering guidelines
    and expert advice on semantic interoperability for public administrations.

    This style guide is intended primarily for:
    - Semantic engineers
    - Data architects
    - Knowledge modelling specialists
    who are acting as editors or reusers of Core Vocabularies and Application Profiles.

    This style guide may also constitute a good source of information and explanations for:
    - European Commission's officers
    - Collaborating consultants
    - Stakeholders involved in interinstitutional standardisation

    Args:
        None

    Returns:
        str: Series of conventions in string format.
"""
