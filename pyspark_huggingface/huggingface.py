from pyspark.sql.datasource import DataSource, DataSourceReader
from pyspark.sql.pandas.types import from_arrow_schema
from pyspark.sql.types import StructType

class HuggingFaceDatasets(DataSource):
    """
    A DataSource for reading and writing HuggingFace Datasets in Spark.

    This data source allows reading public datasets from the HuggingFace Hub directly into Spark
    DataFrames. The schema is automatically inferred from the dataset features. The split can be
    specified using the `split` option. The default split is `train`.

    Name: `huggingface`

    Notes:
    -----
    - The HuggingFace `datasets` library is required to use this data source. Make sure it is installed.
    - If the schema is automatically inferred, it will use string type for all fields.
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
        ds_builder = load_dataset_builder(dataset_name)
        features = ds_builder.info.features
        if features is None:
            raise Exception(
                "Unable to automatically determine the schema using the dataset features. "
                "Please specify the schema manually using `.schema()`."
            )
        return from_arrow_schema(features.arrow_schema)

    def reader(self, schema: StructType) -> "DataSourceReader":
        return HuggingFaceDatasetsReader(schema, self.options)


class HuggingFaceDatasetsReader(DataSourceReader):
    DEFAULT_SPLIT: str = "train"

    def __init__(self, schema: StructType, options: dict):
        self.schema = schema
        self.dataset_name = options["path"]
        # Get and validate the split name
        self.split = options.get("split", self.DEFAULT_SPLIT)
        from datasets import get_dataset_split_names
        valid_splits = get_dataset_split_names(self.dataset_name)
        if self.split not in valid_splits:
            raise Exception(f"Split {self.split} is invalid. Valid options are {valid_splits}")

    def read(self, partition):
        from datasets import load_dataset
        columns = [field.name for field in self.schema.fields]
        # TODO: add config
        iter_dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        for data in iter_dataset:
            # TODO: next spark 4.0.0 dev release will include the feature to yield as an iterator of pa.RecordBatch
            yield tuple([data.get(column) for column in columns])
