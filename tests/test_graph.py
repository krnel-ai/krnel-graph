import pytest
from krnel.graph import ComputationSpec


class ExampleComputationSpec(ComputationSpec):
    """Example computation spec for testing."""
    name: str
    value: int


def test_computation_spec_creation():
    """Test that we can create a ComputationSpec instance."""
    spec = ExampleComputationSpec(name="test", value=42)
    assert spec.name == "test"
    assert spec.value == 42


def test_computation_spec_immutability():
    """Test that ComputationSpec instances are immutable."""
    spec = ExampleComputationSpec(name="test", value=42)
    
    with pytest.raises(Exception):  # Should raise ValidationError or similar
        spec.name = "changed"


def test_computation_spec_content_hash():
    """Test that ComputationSpec generates a content hash."""
    spec = ExampleComputationSpec(name="test", value=42)
    hash_value = spec.content_hash
    
    # Currently returns "HASH HERE" as placeholder
    assert hash_value == "HASH HERE"
    
    # Test that hash is consistent
    assert spec.content_hash == hash_value


def test_computation_spec_hash_consistency():
    """Test that identical specs have the same hash."""
    spec1 = ExampleComputationSpec(name="test", value=42)
    spec2 = ExampleComputationSpec(name="test", value=42)
    
    assert spec1.content_hash == spec2.content_hash


def test_computation_spec_serialization():
    """Test ComputationSpec serialization behavior."""
    spec = ExampleComputationSpec(name="test", value=42)
    
    # Test normal serialization
    normal_dict = spec.model_dump()
    assert normal_dict == {"name": "test", "value": 42}
    
    # Test serialization with hash context
    hash_dict = spec.model_dump(context={"for_hash": True})
    # Should return the content hash when serializing for hash
    assert hash_dict == {"name": "test", "value": 42}  # May change when hash serialization is implemented


if __name__ == "__main__":
    # Interactive testing section
    print("=== Interactive ComputationSpec Testing ===")
    
    # Create test instances
    spec1 = ExampleComputationSpec(name="example1", value=100)
    spec2 = ExampleComputationSpec(name="example2", value=200)
    spec3 = ExampleComputationSpec(name="example1", value=100)  # Same as spec1
    
    print(f"spec1: {spec1}")
    print(f"spec2: {spec2}")
    print(f"spec3: {spec3}")
    
    print(f"\nContent hashes:")
    print(f"spec1.content_hash: {spec1.content_hash}")
    print(f"spec2.content_hash: {spec2.content_hash}")
    print(f"spec3.content_hash: {spec3.content_hash}")
    
    print(f"\nHash equality:")
    print(f"spec1.content_hash == spec3.content_hash: {spec1.content_hash == spec3.content_hash}")
    print(f"spec1.content_hash == spec2.content_hash: {spec1.content_hash == spec2.content_hash}")
    
    print(f"\nSerialization:")
    print(f"spec1.model_dump(): {spec1.model_dump()}")
    print(f"spec1.model_dump(context={{'for_hash': True}}): {spec1.model_dump(context={'for_hash': True})}")