import pytest
from pyspark.sql import SparkSession
from pyspark_huggingface import HuggingFaceDatasets


@pytest.fixture
def spark():
    spark = SparkSession.builder.getOrCreate()
    yield spark


def test_basic_load(spark):
    spark.dataSource.register(HuggingFaceDatasets)
    df = spark.read.format("huggingface").load("rotten_tomatoes")
    assert df.count() == 8530  # length of the training dataset
