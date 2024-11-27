from dataclasses import dataclass
from typing import Sequence

from pyspark.sql.datasource import DataSource, DataSourceReader, InputPartition
from pyspark.sql.pandas.types import from_arrow_schema
from pyspark.sql.types import StructType

class HuggingFaceDatasets(DataSource):
    """
    A DataSource for reading and writing HuggingFace Datasets in Spark.

    This data source allows reading public datasets from the HuggingFace Hub directly into Spark
    DataFrames. The schema is automatically inferred from the dataset features. The split can be
    specified using the `split` option. The default split is `train`.

    Name: `huggingface`

    Data Source Options:
    - split (str): Specify which split to retrieve. Default: train
    - config (str): Specify which subset or configuration to retrieve.
    - streaming (bool): Specify whether to read a dataset without downloading it.

    Notes:
    -----
    - Currently it can only be used with public datasets. Private or gated ones are not supported.

    Examples
    --------

    Load a public dataset from the HuggingFace Hub.

    >>> df = spark.read.format("huggingface").load("imdb")
    DataFrame[text: string, label: bigint]

    >>> df.show()
    +--------------------+-----+
    |                text|label|
    +--------------------+-----+
    |I rented I AM CUR...|    0|
    |"I Am Curious: Ye...|    0|
    |...                 |  ...|
    +--------------------+-----+

    Load a specific split from a public dataset from the HuggingFace Hub.

    >>> spark.read.format("huggingface").option("split", "test").load("imdb").show()
    +--------------------+-----+
    |                text|label|
    +--------------------+-----+
    |I love sci-fi and...|    0|
    |Worth the enterta...|    0|
    |...                 |  ...|
    +--------------------+-----+
    """

    def __init__(self, options):
        super().__init__(options)
        if "path" not in options or not options["path"]:
            raise Exception("You must specify a dataset name.")

    @classmethod
    def name(cls):
        return "huggingface"

    def schema(self):
        from datasets import load_dataset_builder
        dataset_name = self.options["path"]
        config_name = self.options.get("config")
        ds_builder = load_dataset_builder(dataset_name, config_name)
        features = ds_builder.info.features
        if features is None:
            raise Exception(
                "Unable to automatically determine the schema using the dataset features. "
                "Please specify the schema manually using `.schema()`."
            )
        return from_arrow_schema(features.arrow_schema)

    def reader(self, schema: StructType) -> "DataSourceReader":
        return HuggingFaceDatasetsReader(schema, self.options)


@dataclass
class Shard(InputPartition):
    """ Represents a dataset shard. """
    index: int


class HuggingFaceDatasetsReader(DataSourceReader):
    DEFAULT_SPLIT: str = "train"

    def __init__(self, schema: StructType, options: dict):
        from datasets import get_dataset_split_names, get_dataset_default_config_name
        self.schema = schema
        self.dataset_name = options["path"]
        self.streaming = options.get("streaming", "true").lower() == "true"
        self.config_name = options.get("config")
        # Get and validate the split name
        self.split = options.get("split", self.DEFAULT_SPLIT)
        valid_splits = get_dataset_split_names(self.dataset_name, self.config_name)
        if self.split not in valid_splits:
            raise Exception(f"Split {self.split} is invalid. Valid options are {valid_splits}")

    def partitions(self) -> Sequence[InputPartition]:
        from datasets import load_dataset
        if not self.streaming:
            return [Shard(index=0)]
        else:
            dataset = load_dataset(self.dataset_name, name=self.config_name, split=self.split, streaming=True)
            return [Shard(index=i) for i in range(dataset.num_shards)]

    def read(self, partition: Shard):
        from datasets import load_dataset
        columns = [field.name for field in self.schema.fields]
        dataset = load_dataset(self.dataset_name, name=self.config_name, split=self.split, streaming=self.streaming)
        if self.streaming:
            shard = dataset.shard(num_shards=dataset.num_shards, index=partition.index)
            for _, pa_table in shard._ex_iterable.iter_arrow():
                yield from pa_table.select(columns).to_batches()
        else:
            # Get the underlying arrow table of the dataset
            table = dataset._data
            yield from table.select(columns).to_batches()
