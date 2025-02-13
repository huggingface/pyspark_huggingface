from typing import TYPE_CHECKING, Optional

from pyspark.sql.datasource import DataSource

if TYPE_CHECKING:
    from pyspark.sql.datasource import DataSourceArrowWriter, DataSourceReader
    from pyspark.sql.types import StructType

    from pyspark_huggingface.huggingface_sink import HuggingFaceSink
    from pyspark_huggingface.huggingface_source import HuggingFaceSource


class HuggingFaceDatasets(DataSource):
    """
    DataSource for reading and writing HuggingFace Datasets in Spark.

    Read
    ------
    See :py:class:`HuggingFaceSource` for more details.

    Write
    ------
    See :py:class:`HuggingFaceSink` for more details.
    """

    # Delegate the source and sink methods to the respective classes.
    def __init__(self, options: dict):
        super().__init__(options)
        self.options = options
        self.source: Optional["HuggingFaceSource"] = None
        self.sink: Optional["HuggingFaceSink"] = None

    def get_source(self) -> "HuggingFaceSource":
        from pyspark_huggingface.huggingface_source import HuggingFaceSource

        if self.source is None:
            self.source = HuggingFaceSource(self.options.copy())
        return self.source

    def get_sink(self) -> "HuggingFaceSink":
        from pyspark_huggingface.huggingface_sink import HuggingFaceSink

        if self.sink is None:
            self.sink = HuggingFaceSink(self.options.copy())
        return self.sink

    @classmethod
    def name(cls):
        return "huggingface"

    def schema(self):
        return self.get_source().schema()

    def reader(self, schema: "StructType") -> "DataSourceReader":
        return self.get_source().reader(schema)

    def writer(self, schema: "StructType", overwrite: bool) -> "DataSourceArrowWriter":
        return self.get_sink().writer(schema, overwrite)
