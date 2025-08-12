<p align="center">
  <img alt="Hugging Face x Spark" src="https://pbs.twimg.com/media/FvN1b_2XwAAWI1H?format=jpg&name=large" width="352" style="max-width: 100%;">
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://github.com/huggingface/pyspark_huggingface/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/pyspark_huggingface.svg"></a>
    <a href="https://huggingface.co/datasets/"><img alt="Number of datasets" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/datasets&color=brightgreen"></a>
</p>

# Spark Data Source for Hugging Face Datasets

A Spark Data Source for accessing [🤗 Hugging Face Datasets](https://huggingface.co/datasets):

- Stream datasets from Hugging Face as Spark DataFrames
- Select subsets and splits, apply projection and predicate filters
- Save Spark DataFrames as Parquet files to Hugging Face
- Fast deduped uploads
- Fully distributed
- Authentication via `huggingface-cli login` or tokens
- Compatible with Spark 4 (with auto-import)
- Backport for Spark 3.5, 3.4 and 3.3

## Installation

```
pip install pyspark_huggingface
```

## Usage

Load a dataset (here [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)):

```python
import pyspark_huggingface
df = spark.read.format("huggingface").load("stanfordnlp/imdb")
```

Save to Hugging Face:

```python
# Login with huggingface-cli login
df.write.format("huggingface").save("username/my_dataset")
# Or pass a token manually
df.write.format("huggingface").option("token", "hf_xxx").save("username/my_dataset")
```

## Advanced

Select a split:

```python
test_df = (
    spark.read.format("huggingface")
    .option("split", "test")
    .load("stanfordnlp/imdb")
)
```

Select a subset/config:

```python
test_df = (
    spark.read.format("huggingface")
    .option("config", "sample-10BT")
    .load("HuggingFaceFW/fineweb-edu")
)
```

Filters columns and rows (especially efficient for Parquet datasets):

```python
df = (
    spark.read.format("huggingface")
    .option("filters", '[("language_score", ">", 0.99)]')
    .option("columns", '["text", "language_score"]')
    .load("HuggingFaceFW/fineweb-edu")
)
```

## Fast deduped uploads

Hugging Face uses Xet: a dedupe-based storage which enables fast deduped uploads.

Unlike traditional remote storage, uploads are faster on Xet because duplicate data is only uploaded once.
For example: if some or all of the data already exists in other files on Xet, it is not uploaded again, saving bandwidth and speeding up uploads. Deduplication for Parquet is enabled through Content Defined Chunking (CDC).

Thanks to Parquet CDC and Xet deduplication, saving a dataset on Hugging Face is faster than on any traditional remote storage.

For more information, see [https://huggingface.co/blog/parquet-cdc](https://huggingface.co/blog/parquet-cdc).

## Backport

While the Data Source API was introcuded in Spark 4, this package includes a backport for older versions.

Importing `pyspark_huggingface` patches the PySpark reader and writer to add the "huggingface" data source. It is compatible with PySpark 3.5, 3.4 and 3.3:

```python
>>> import pyspark_huggingface
huggingface datasource enabled for pyspark 3.x.x (backport from pyspark 4)
```

The import is only necessary on Spark 3.x to enable the backport.
Spark 4 automatically imports `pyspark_huggingface` as soon as it is installed, and registers the "huggingface" data source.


## Development

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if not already done.

Then, from the project root directory, sync dependencies and run tests.
```
uv sync
uv run pytest
```
