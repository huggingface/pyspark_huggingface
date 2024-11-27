import pytest
from pyspark.sql import SparkSession

@pytest.fixture
def spark():
    spark = SparkSession.builder.getOrCreate()
    yield spark


def test_basic_load(spark):
    df = spark.read.format("huggingface").load("rotten_tomatoes")
    assert df.count() == 8530  # length of the training dataset
