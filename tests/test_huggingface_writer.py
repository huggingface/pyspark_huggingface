import os
import uuid

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.testing import assertDataFrameEqual
from pytest_mock import MockerFixture

import pyspark_huggingface  # noqa: F401

# ============== Fixtures & Helpers ==============


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.getOrCreate()
    yield spark


def token():
    if "HF_TOKEN" not in os.environ:
        pytest.skip("HF_TOKEN environment variable is not set")
    return os.environ["HF_TOKEN"]


def load(repo, split):
    from datasets import load_dataset

    return load_dataset(repo, token=token(), split=split).to_pandas()


def writer(df: DataFrame):
    return df.write.format("huggingface").option("token", token())


@pytest.fixture(scope="session")
def random_df(spark: SparkSession):
    from pyspark.sql.functions import rand

    return lambda n: spark.range(n, numPartitions=2).select((rand()).alias("value"))


@pytest.fixture(scope="session")
def api():
    import huggingface_hub

    return huggingface_hub.HfApi(token=token())


@pytest.fixture(scope="session")
def username(api):
    return api.whoami()["name"]


@pytest.fixture
def repo(api, username):
    repo_id = f"{username}/test-{uuid.uuid4()}"
    api.create_repo(repo_id, private=False, repo_type="dataset")
    yield repo_id
    api.delete_repo(repo_id, repo_type="dataset")


# ============== Tests ==============


def test_basic(repo, random_df):
    df = random_df(10)
    writer(df).mode("append").save(repo)
    actual = load(repo, "train")
    assertDataFrameEqual(actual, df.toPandas())


@pytest.mark.parametrize("split", ["train", "custom"])
def test_append(repo, random_df, split):
    df1 = random_df(10)
    df2 = random_df(10)
    writer(df1).options(split=split).mode("append").save(repo)
    writer(df2).options(split=split).mode("append").save(repo)
    actual = load(repo, split)
    expected = df1.union(df2)
    assertDataFrameEqual(actual, expected.toPandas())


@pytest.mark.parametrize("split", ["train", "custom"])
def test_overwrite(repo, random_df, split):
    df1 = random_df(10)
    df2 = random_df(10)
    writer(df1).options(split=split).mode("append").save(repo)
    writer(df2).options(split=split).mode("overwrite").save(repo)
    actual = load(repo, split)
    assertDataFrameEqual(actual, df2.toPandas())


def test_split(repo, random_df):
    df1 = random_df(10)
    df2 = random_df(10)
    writer(df1).mode("append").save(repo)
    writer(df2).mode("append").options(split="custom").save(repo)
    actual1 = load(repo, "train")
    actual2 = load(repo, "custom")
    assertDataFrameEqual(actual1, df1.toPandas())
    assertDataFrameEqual(actual2, df2.toPandas())


def test_revision(repo, random_df, api):
    df = random_df(10)
    api.create_branch(repo, branch="test", repo_type="dataset")
    writer(df).mode("append").options(revision="test").save(repo)
    assert any(
        file.path.endswith(".parquet")
        for file in api.list_repo_tree(
            repo, repo_type="dataset", revision="test", recursive=True
        )
    )


def test_max_bytes_per_file(spark, mocker: MockerFixture):
    from pyspark_huggingface.huggingface_sink import HuggingFaceDatasetsWriter

    repo = "user/test"
    api = mocker.patch("huggingface_hub.HfApi").return_value = mocker.MagicMock()

    df = spark.range(10)
    writer = HuggingFaceDatasetsWriter(
        repo_id=repo,
        overwrite=False,
        schema=df.schema,
        token="token",
        max_bytes_per_file=1,
    )
    writer.write(iter(df.toArrow().to_batches(max_chunksize=1)))
    assert api.preupload_lfs_files.call_count == 10
