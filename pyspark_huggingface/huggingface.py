from typing import Optional

from pyspark.sql.datasource import DataSource, DataSourceArrowWriter, DataSourceReader
from pyspark.sql.types import StructType

from pyspark_huggingface.huggingface_sink import HuggingFaceSink
from pyspark_huggingface.huggingface_source import HuggingFaceSource


class HuggingFaceDatasets(DataSource):
    def __init__(self, options: dict):
        super().__init__(options)
        self.options = options
        self.source: Optional[HuggingFaceSource] = None
        self.sink: Optional[HuggingFaceSink] = None

    def get_source(self) -> HuggingFaceSource:
        assert (
            self.sink is None
        ), "Cannot read and write from the same data source instance"
        if self.source is None:
            self.source = HuggingFaceSource(self.options)
        return self.source

    def get_sink(self):
        assert (
            self.source is None
        ), "Cannot read and write from the same data source instance"
        if self.sink is None:
            self.sink = HuggingFaceSink(self.options)
        return self.sink

    @classmethod
    def name(cls):
        return "huggingface"

    def schema(self):
        return self.get_source().schema()

    def reader(self, schema: StructType) -> "DataSourceReader":
        return self.get_source().reader(schema)

    def writer(self, schema: StructType, overwrite: bool) -> "DataSourceArrowWriter":
        return self.get_sink().writer(schema, overwrite)
