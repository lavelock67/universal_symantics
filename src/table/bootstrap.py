"""Bootstrap helpers to augment a periodic table with provisional basics.

These primitives are added only if missing. They are marked with simple
signatures and algebraic flags to enable early experiments (detectors/MDL).
"""

from .schema import PeriodicTable, Primitive, PrimitiveCategory, PrimitiveSignature


def augment_with_basics(table: PeriodicTable) -> None:
    basics = [
        Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2), transitive=True, antisymmetric=True, description="Bootstrap IsA"),
        Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2), transitive=True, antisymmetric=True, description="Bootstrap PartOf"),
        Primitive("AtLocation", PrimitiveCategory.SPATIAL, PrimitiveSignature(arity=2), antisymmetric=True, description="Bootstrap AtLocation"),
        Primitive("Before", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2), description="Bootstrap Before"),
        Primitive("After", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2), description="Bootstrap After"),
        Primitive("SimilarTo", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2), symmetric=True, description="Bootstrap SimilarTo"),
        Primitive("Antonym", PrimitiveCategory.LOGICAL, PrimitiveSignature(arity=2), symmetric=True, description="Bootstrap Antonym"),
    ]
    for p in basics:
        if table.get_primitive(p.name) is None:
            try:
                table.add_primitive(p)
            except Exception:
                pass


