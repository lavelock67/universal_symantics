"""Seed the periodic table with Natural Semantic Metalanguage (NSM) style primes.

This is a pragmatic subset to broaden coverage for early experiments.
"""

from typing import List, Tuple

from .schema import PeriodicTable, Primitive, PrimitiveCategory, PrimitiveSignature


NSM_PRIMES: List[Tuple[str, PrimitiveCategory, int, dict]] = [
    # Existentials / identity / part-whole
    ("PartOf", PrimitiveCategory.STRUCTURAL, 2, {"transitive": True, "antisymmetric": True}),
    ("SameAs", PrimitiveCategory.LOGICAL, 2, {}),
    ("OtherThan", PrimitiveCategory.LOGICAL, 2, {}),
    ("KindOf", PrimitiveCategory.INFORMATIONAL, 2, {"transitive": True, "antisymmetric": True}),
    # Spatial / place
    ("In", PrimitiveCategory.SPATIAL, 2, {}),
    ("On", PrimitiveCategory.SPATIAL, 2, {}),
    ("At", PrimitiveCategory.SPATIAL, 2, {}),
    ("Near", PrimitiveCategory.SPATIAL, 2, {}),
    # Temporal / order
    ("Before", PrimitiveCategory.TEMPORAL, 2, {}),
    ("After", PrimitiveCategory.TEMPORAL, 2, {}),
    ("During", PrimitiveCategory.TEMPORAL, 2, {}),
    ("When", PrimitiveCategory.TEMPORAL, 2, {}),
    ("While", PrimitiveCategory.TEMPORAL, 2, {}),
    # Quality / evaluation
    ("Good", PrimitiveCategory.COGNITIVE, 1, {}),
    ("Bad", PrimitiveCategory.COGNITIVE, 1, {}),
    ("Big", PrimitiveCategory.QUANTITATIVE, 1, {}),
    ("Small", PrimitiveCategory.QUANTITATIVE, 1, {}),
    # Quantifiers / comparison
    ("All", PrimitiveCategory.QUANTITATIVE, 1, {}),
    ("Some", PrimitiveCategory.QUANTITATIVE, 1, {}),
    ("Many", PrimitiveCategory.QUANTITATIVE, 1, {}),
    ("Few", PrimitiveCategory.QUANTITATIVE, 1, {}),
    ("MoreThan", PrimitiveCategory.QUANTITATIVE, 2, {}),
    ("LessThan", PrimitiveCategory.QUANTITATIVE, 2, {}),
    ("EqualTo", PrimitiveCategory.QUANTITATIVE, 2, {}),
    ("Most", PrimitiveCategory.QUANTITATIVE, 1, {}),
    ("None", PrimitiveCategory.QUANTITATIVE, 1, {}),
    # Modality / polarity
    ("Not", PrimitiveCategory.LOGICAL, 1, {}),
    ("Can", PrimitiveCategory.COGNITIVE, 2, {}),
    ("Must", PrimitiveCategory.COGNITIVE, 2, {}),
    # Mental predicates
    ("Think", PrimitiveCategory.COGNITIVE, 2, {}),
    ("Know", PrimitiveCategory.COGNITIVE, 2, {}),
    ("Want", PrimitiveCategory.COGNITIVE, 2, {}),
    ("Feel", PrimitiveCategory.COGNITIVE, 1, {}),
    ("Say", PrimitiveCategory.COGNITIVE, 2, {}),
    # Actions / motion / perception
    ("Do", PrimitiveCategory.INFORMATIONAL, 2, {}),
    ("Go", PrimitiveCategory.SPATIAL, 2, {}),
    ("Move", PrimitiveCategory.SPATIAL, 2, {}),
    ("See", PrimitiveCategory.COGNITIVE, 2, {}),
    ("Hear", PrimitiveCategory.COGNITIVE, 2, {}),
    # Causal / purpose / conditionals / aspect
    ("Because", PrimitiveCategory.CAUSAL, 2, {}),
    ("Therefore", PrimitiveCategory.CAUSAL, 2, {}),
    ("If", PrimitiveCategory.LOGICAL, 2, {}),
    ("Unless", PrimitiveCategory.LOGICAL, 2, {}),
    ("InOrderTo", PrimitiveCategory.COGNITIVE, 2, {}),
    ("Progressive", PrimitiveCategory.TEMPORAL, 1, {}),
    ("Perfect", PrimitiveCategory.TEMPORAL, 1, {}),
]


def augment_with_nsm_primes(table: PeriodicTable) -> None:
    for name, cat, arity, flags in NSM_PRIMES:
        if table.get_primitive(name):
            continue
        prim = Primitive(
            name=name,
            category=cat,
            signature=PrimitiveSignature(arity=arity),
            symmetric=flags.get("symmetric", False),
            transitive=flags.get("transitive", False),
            antisymmetric=flags.get("antisymmetric", False),
            description=f"NSM seed: {name}",
        )
        try:
            table.add_primitive(prim)
        except Exception:
            pass


