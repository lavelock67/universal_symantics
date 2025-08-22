"""Simple text generators for primitives.

These are minimal, human-readable templates to sanity-check that primitives
can be rendered as language. Not used for training or evaluation.
"""

from typing import Dict, List


_TEMPLATES: Dict[str, str] = {
    "IsA": "{x} is a {y}",
    "PartOf": "{x} is part of {y}",
    "AtLocation": "{x} is at the {y}",
    "Before": "{x} happens before {y}",
    "After": "{x} happens after {y}",
    "UsedFor": "{x} is used for {y}",
    "CapableOf": "{x} is capable of {y}",
    "Antonym": "{x} is the opposite of {y}",
    "SimilarTo": "{x} is similar to {y}",
    "RelatedTo": "{x} is related to {y}",
}


def generate_sentence(primitive_name: str, arguments: List[str]) -> str:
    """Generate a simple sentence for a primitive with two arguments.

    Falls back to a generic pattern if no template exists.
    """
    tpl = _TEMPLATES.get(primitive_name, "{x} {rel} {y}")
    x = arguments[0] if len(arguments) > 0 else "X"
    y = arguments[1] if len(arguments) > 1 else "Y"
    if "{rel}" in tpl:
        return tpl.format(x=x, y=y, rel=primitive_name.lower())
    return tpl.format(x=x, y=y)


