import pytest
from krnel.graph import ComputationSpec


class ExampleDataSource(ComputationSpec):
    """Example computation spec for testing."""
    dataset_name: str
    import_date: str

class DatasetDoubleSize(ComputationSpec):
    """Operation that doubles the size of the dataset!"""
    power_level: str = "DOUBLE"
    source_dataset: ExampleDataSource

def test_computation_spec_immutability():
    """Test that ComputationSpec instances are immutable."""
    spec = ExampleDataSource(dataset_name="test", import_date="2023-01-01")
    assert spec.dataset_name == "test"
    assert spec.import_date == "2023-01-01"

    with pytest.raises(Exception):  # Should raise ValidationError or similar
        spec.dataset_name = "changed"


def test_computation_spec_content_hash():
    """Test that ComputationSpec generates a content hash."""
    spec = ExampleDataSource(dataset_name="test", import_date="2023-01-01")

    assert spec.content_hash == "step-QFl6e_7VoThY7n_lAwcX1F9CNG9ctDuoEpq4cSpWzJs"


def test_computation_spec_hash_consistency():
    """Test that identical specs have the same hash."""
    spec1 = ExampleDataSource(dataset_name="test", import_date="2023-01-01")
    spec2 = ExampleDataSource(dataset_name="test", import_date="2023-01-01")
    spec3 = ExampleDataSource(dataset_name="bar", import_date="2023-01-01")

    assert spec1.content_hash == spec2.content_hash
    assert spec3.content_hash != spec1.content_hash


def test_computation_spec_serialization():
    """Test ComputationSpec serialization behavior."""
    data = ExampleDataSource(dataset_name="test", import_date="2023-01-01")
    double_op = DatasetDoubleSize(source_dataset=data)

    # Test normal serialization
    normal_dict = double_op.model_dump()
    assert normal_dict == {
        "source_dataset": {
            "dataset_name": "test",
            "import_date": "2023-01-01",
        },
        "power_level": "DOUBLE",
    }

    # Test serialization with hash context
    hash_dict = double_op.model_dump(context={"for_hash": True})
    assert hash_dict == {
        "source_dataset": "step-QFl6e_7VoThY7n_lAwcX1F9CNG9ctDuoEpq4cSpWzJs",
        "power_level": "DOUBLE",
    }


def test_serialization_for_hash_false():
    """Test serialization when for_hash is False."""
    data = ExampleDataSource(dataset_name="test", import_date="2023-01-01")
    result = data.model_dump(context={"for_hash": False})
    assert result == {"dataset_name": "test", "import_date": "2023-01-01"}



def test_bad():
    assert False