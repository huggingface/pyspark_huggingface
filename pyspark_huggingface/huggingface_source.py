import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from pyspark.sql.datasource import DataSource, DataSourceReader, InputPartition
from pyspark.sql.pandas.types import from_arrow_schema
from pyspark.sql.types import StructType

if TYPE_CHECKING:
    from datasets import DatasetBuilder, IterableDataset

class HuggingFaceSource(DataSource):
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

    Enable predicate pushdown for Parquet datasets.

    >>> spark.read.format("huggingface") \
    ...     .option("filters", '[("language_score", ">", 0.99)]') \
    ...     .option("columns", '["text", "language_score"]') \
    ...     .load("HuggingFaceFW/fineweb-edu") \
    ...     .show()
    +--------------------+------------------+                                       
    |                text|    language_score|
    +--------------------+------------------+
    |died Aug. 28, 181...|0.9901925325393677|
    |Coyotes spend a g...|0.9902171492576599|
    |...                 |               ...|
    +--------------------+------------------+
    """

    DEFAULT_SPLIT: str = "train"

    def __init__(self, options):
        super().__init__(options)
        from datasets import load_dataset_builder

        if "path" not in options or not options["path"]:
            raise Exception("You must specify a dataset name.")

        kwargs = dict(self.options)
        self.dataset_name = kwargs.pop("path")
        self.config_name = kwargs.pop("config", None)
        self.split = kwargs.pop("split", self.DEFAULT_SPLIT)
        self.streaming = kwargs.pop("streaming", "true").lower() == "true"
        for arg in kwargs:
            if kwargs[arg].lower() == "true":
                kwargs[arg] = True
            elif kwargs[arg].lower() == "false":
                kwargs[arg] = False
            else:
                try:
                    kwargs[arg] = ast.literal_eval(kwargs[arg])
                except ValueError:
                    pass

        self.builder = load_dataset_builder(self.dataset_name, self.config_name, **kwargs)
        streaming_dataset = self.builder.as_streaming_dataset()
        if self.split not in streaming_dataset:
            raise Exception(f"Split {self.split} is invalid. Valid options are {list(streaming_dataset)}")

        self.streaming_dataset = streaming_dataset[self.split]
        if not self.streaming_dataset.features:
            self.streaming_dataset = self.streaming_dataset._resolve_features()

    @classmethod
    def name(cls):
        return "huggingfacesource"

    def schema(self):
        return from_arrow_schema(self.streaming_dataset.features.arrow_schema)

    def reader(self, schema: StructType) -> "DataSourceReader":
        return HuggingFaceDatasetsReader(
            schema,
            builder=self.builder,
            split=self.split,
            streaming_dataset=self.streaming_dataset if self.streaming else None
        )


@dataclass
class Shard(InputPartition):
    """ Represents a dataset shard. """

    index: int


class HuggingFaceDatasetsReader(DataSourceReader):
    def __init__(self, schema: StructType, builder: "DatasetBuilder", split: str, streaming_dataset: Optional["IterableDataset"]):
        self.schema = schema
        self.builder = builder
        self.split = split
        self.streaming_dataset = streaming_dataset
        # Get and validate the split name

    def partitions(self) -> Sequence[InputPartition]:
        if self.streaming_dataset:
            return [Shard(index=i) for i in range(self.streaming_dataset.num_shards)]
        else:
            return [Shard(index=0)]

    def read(self, partition: Shard):
        columns = [field.name for field in self.schema.fields]
        if self.streaming_dataset:
            shard = self.streaming_dataset.shard(num_shards=self.streaming_dataset.num_shards, index=partition.index)
            if shard._ex_iterable.iter_arrow:
                for _, pa_table in shard._ex_iterable.iter_arrow():
                    yield from pa_table.select(columns).to_batches()
            else:
                for _, example in shard:
                    yield example
        else:
            self.builder.download_and_prepare()
            dataset = self.builder.as_dataset(self.split)
            # Get the underlying arrow table of the dataset
            table = dataset._data
            yield from table.select(columns).to_batches()
