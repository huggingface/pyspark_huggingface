import os

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.testing import assertDataFrameEqual
from pytest_mock import MockerFixture


# ============== Fixtures & Helpers ==============

@pytest.fixture(scope="session")
def spark():
    from pyspark_huggingface.huggingface_sink import HuggingFaceSink

    spark = SparkSession.builder.getOrCreate()
    spark.dataSource.register(HuggingFaceSink)
    yield spark


def token():
    return os.environ["HF_TOKEN"]


def reader(spark):
    return spark.read.format("huggingface").option("token", token())


def writer(df: DataFrame):
    return df.write.format("huggingfacesink").option("token", token())


@pytest.fixture(scope="session")
def random_df(spark: SparkSession):
    from pyspark.sql.functions import rand

    return lambda n: spark.range(n).select((rand()).alias("value"))


@pytest.fixture(scope="session")
def api():
    import huggingface_hub

    return huggingface_hub.HfApi(token=token())


@pytest.fixture(scope="session")
def username(api):
    return api.whoami()["name"]


@pytest.fixture
def repo(api, username):
    import uuid

    repo_id = f"{username}/test-{uuid.uuid4()}"
    api.create_repo(repo_id, private=True, repo_type="dataset")
    yield repo_id
    api.delete_repo(repo_id, repo_type="dataset")


# ============== Tests ==============

def test_basic(spark, repo, random_df):
    df = random_df(10)
    writer(df).mode("append").save(repo)
    actual = reader(spark).load(repo)
    assertDataFrameEqual(df, actual)


def test_append(spark, repo, random_df):
    df1 = random_df(10)
    df2 = random_df(10)
    writer(df1).mode("append").save(repo)
    writer(df2).mode("append").save(repo)
    actual = reader(spark).load(repo)
    expected = df1.union(df2)
    assertDataFrameEqual(actual, expected)


def test_overwrite(spark, repo, random_df):
    df1 = random_df(10)
    df2 = random_df(10)
    writer(df1).mode("append").save(repo)
    writer(df2).mode("overwrite").save(repo)
    actual = reader(spark).load(repo)
    assertDataFrameEqual(actual, df2)


def test_dir(repo, random_df, api):
    df = random_df(10)
    writer(df).mode("append").save(repo + "/dir")
    assert any(
        file.path.endswith(".parquet")
        for file in api.list_repo_tree(repo, "dir", repo_type="dataset")
    )


def test_revision(repo, random_df, api):
    df = random_df(10)
    api.create_branch(repo, branch="test", repo_type="dataset")
    writer(df).mode("append").save(repo + "@test")
    assert any(
        file.path.endswith(".parquet")
        for file in api.list_repo_tree(repo, repo_type="dataset", revision="test")
    )


def test_max_bytes_per_file(spark, mocker: MockerFixture):
    from pyspark_huggingface.huggingface_sink import HuggingFaceDatasetsWriter

    repo = "user/test"
    fs = mocker.patch("huggingface_hub.HfFileSystem").return_value = mocker.MagicMock()
    resolved_path = fs.resolve_path.return_value = mocker.MagicMock()
    resolved_path.path_in_repo = repo
    resolved_path.repo_id = repo
    resolved_path.repo_type = "dataset"
    resolved_path.revision = "main"

    df = spark.range(10)
    writer = HuggingFaceDatasetsWriter(
        path=repo,
        overwrite=False,
        schema=df.schema,
        token="token",
        max_bytes_per_file=1,
    )
    writer.write(iter(df.toArrow().to_batches(max_chunksize=1)))
    assert fs._api.preupload_lfs_files.call_count == 10
