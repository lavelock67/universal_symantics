"""Basic tests for the periodic primitives system."""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.table import (
    Primitive, 
    PrimitiveCategory, 
    PrimitiveSignature, 
    PeriodicTable,
    PrimitiveAlgebra
)
from src.specialists import TemporalESNSpecialist, CentralHub
from src.validate import CompressionValidator


class TestBasicFunctionality:
    """Test basic functionality of the periodic primitives system."""
    
    def test_primitive_creation(self):
        """Test creating primitives."""
        # Create a simple primitive
        signature = PrimitiveSignature(arity=2)
        primitive = Primitive(
            name="IsA",
            category=PrimitiveCategory.INFORMATIONAL,
            signature=signature,
            description="Is-a relationship",
            examples=["cat is animal", "car is vehicle"]
        )
        
        assert primitive.name == "IsA"
        assert primitive.category == PrimitiveCategory.INFORMATIONAL
        assert primitive.arity == 2
        assert len(primitive.examples) == 2
    
    def test_periodic_table(self):
        """Test periodic table operations."""
        table = PeriodicTable()
        
        # Add primitives
        primitives = [
            Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2)),
            Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2)),
            Primitive("Before", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2)),
        ]
        
        for primitive in primitives:
            table.add_primitive(primitive)
        
        assert len(table.primitives) == 3
        assert len(table.categories) == 3
        
        # Test getting primitives by category
        info_primitives = table.get_primitives_by_category(PrimitiveCategory.INFORMATIONAL)
        assert len(info_primitives) == 1
        assert info_primitives[0].name == "IsA"
    
    def test_algebra_operations(self):
        """Test primitive algebra operations."""
        table = PeriodicTable()
        
        # Add some primitives
        primitives = [
            Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2)),
            Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2)),
        ]
        
        for primitive in primitives:
            table.add_primitive(primitive)
        
        algebra = PrimitiveAlgebra(table)
        
        # Test factorization
        text = "The cat is on the mat"
        factors = algebra.factor(text)
        assert isinstance(factors, list)
        
        # Test difference operator
        prior = "The cat is on the mat"
        new = "The dog is in the house"
        differences = algebra.delta(prior, new)
        assert isinstance(differences, list)
    
    def test_temporal_esn(self):
        """Test temporal ESN specialist."""
        specialist = TemporalESNSpecialist(reservoir_size=64, n_esns=2)
        
        # Test sequence processing
        sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = specialist.process_sequence(sequence)
        
        assert "temporal_features" in result
        assert "sequence_length" in result
        assert result["sequence_length"] == 5
        
        # Test temporal coherence
        coherence = specialist.compute_temporal_coherence(sequence)
        assert 0.0 <= coherence <= 1.0
    
    def test_central_hub(self):
        """Test central integration hub."""
        table = PeriodicTable()
        table.add_primitive(
            Primitive("Test", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=1))
        )
        
        hub = CentralHub(table)
        
        # Test integration
        signals = {
            "vision": np.array([1.0, 2.0, 3.0]),
            "audio": np.array([0.5, 1.0, 1.5]),
        }
        
        result = hub.integrate(signals)
        
        print(f"Integration result: {result}")
        
        assert "integration_score" in result
        assert "phi_score" in result
        assert "attention_weights" in result
        assert 0.0 <= result["integration_score"] <= 1.0
        # Phi score can be 0 for single modality, so just check it's not negative
        assert result["phi_score"] >= 0.0

    def test_algebraic_properties(self):
        """Verify algebraic flags on common relations when present."""
        table = PeriodicTable()
        # Manually add a few primitives with algebra flags
        table.add_primitive(Primitive("Antonym", PrimitiveCategory.LOGICAL, PrimitiveSignature(arity=2), symmetric=True))
        table.add_primitive(Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2), transitive=True, antisymmetric=True))
        table.add_primitive(Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2), transitive=True, antisymmetric=True))

        antonym = table.get_primitive("Antonym")
        isa = table.get_primitive("IsA")
        partof = table.get_primitive("PartOf")

        assert antonym is not None and antonym.symmetric is True
        assert isa is not None and isa.transitive is True and isa.antisymmetric is True
        assert partof is not None and partof.transitive is True and partof.antisymmetric is True
    
    def test_compression_validator(self):
        """Test compression validation."""
        table = PeriodicTable()
        
        # Add some primitives
        primitives = [
            Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2)),
            Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2)),
            Primitive("Before", PrimitiveCategory.TEMPORAL, PrimitiveSignature(arity=2)),
        ]
        
        for primitive in primitives:
            table.add_primitive(primitive)
        
        validator = CompressionValidator(table)
        
        # Test compression calculation
        data = "The cat is on the mat"
        codebook = list(table.primitives.values())
        
        mdl_score = validator.calculate_mdl_score(data, codebook)
        assert mdl_score > 0
        
        compression_ratio = validator.calculate_compression_ratio(data, codebook)
        assert compression_ratio > 0
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        table = PeriodicTable()
        
        # Add primitives
        primitives = [
            Primitive("IsA", PrimitiveCategory.INFORMATIONAL, PrimitiveSignature(arity=2)),
            Primitive("PartOf", PrimitiveCategory.STRUCTURAL, PrimitiveSignature(arity=2)),
        ]
        
        for primitive in primitives:
            table.add_primitive(primitive)
        
        # Test validation
        errors = table.validate()
        assert len(errors) == 0
        
        # Test serialization
        table_dict = table.to_dict()
        assert "version" in table_dict
        assert "primitives" in table_dict
        assert "categories" in table_dict
        
        # Test deserialization
        reconstructed_table = PeriodicTable.from_dict(table_dict)
        assert len(reconstructed_table.primitives) == len(table.primitives)
        assert len(reconstructed_table.categories) == len(table.categories)
    
    def test_file_io(self):
        """Test file input/output operations."""
        table = PeriodicTable()
        
        # Add a primitive
        primitive = Primitive(
            "TestPrimitive", 
            PrimitiveCategory.INFORMATIONAL, 
            PrimitiveSignature(arity=1)
        )
        table.add_primitive(primitive)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(table.to_dict(), f, indent=2)
            temp_path = f.name
        
        try:
            # Load from file
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            loaded_table = PeriodicTable.from_dict(data)
            assert len(loaded_table.primitives) == 1
            assert "TestPrimitive" in loaded_table.primitives
            
        finally:
            # Clean up
            Path(temp_path).unlink()


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestBasicFunctionality()
    
    print("Running basic functionality tests...")
    
    test_instance.test_primitive_creation()
    print("✅ Primitive creation test passed")
    
    test_instance.test_periodic_table()
    print("✅ Periodic table test passed")
    
    test_instance.test_algebra_operations()
    print("✅ Algebra operations test passed")
    
    test_instance.test_temporal_esn()
    print("✅ Temporal ESN test passed")
    
    test_instance.test_central_hub()
    print("✅ Central hub test passed")
    
    test_instance.test_compression_validator()
    print("✅ Compression validator test passed")
    
    test_instance.test_serialization()
    print("✅ Serialization test passed")
    
    test_instance.test_file_io()
    print("✅ File I/O test passed")
    
    print("\n🎉 All basic tests passed!")
