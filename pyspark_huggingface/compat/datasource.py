from typing import TYPE_CHECKING, Iterator, List, Optional, Union

import pyspark


try:
    from pyspark.sql.datasource import DataSource, DataSourceArrowWriter, DataSourceReader, DataSourceWriter, InputPartition, WriterCommitMessage
except ImportError:
    class DataSource:
        def __init__(self, options):
            self.options = options

    class DataSourceArrowWriter:
        ...

    class DataSourceReader:
        ...
    
    class DataSourceWriter:
        def __init__(self, options):
            self.options = options

    class InputPartition:
        ...
    
    class WriterCommitMessage:
        ...
    

    import logging
    import os
    import pickle
    from functools import wraps

    from pyspark.sql.readwriter import DataFrameReader as _DataFrameReader, DataFrameWriter as _DataFrameWriter

    if TYPE_CHECKING:
        from pyarrow import RecordBatch
        from pyspark.sql.dataframe import DataFrame
        from pyspark.sql.readwriter import PathOrPaths
        from pyspark.sql.types import StructType
        from pyspark.sql._typing import OptionalPrimitiveType


    class _ArrowPickler:

        def __init__(self, key: str):
            from pyspark.sql.types import StructType, StructField, BinaryType

            self.key = key
            self.schema = StructType([StructField(self.key, BinaryType(), True)])
        
        def dumps(self, obj):
            return {self.key: pickle.dumps(obj)}

        def loads(self, obj):
            return pickle.loads(obj[self.key])
    
    # Reader

    def _read_in_arrow(batches: Iterator["RecordBatch"], arrow_pickler, hf_reader) -> Iterator["RecordBatch"]:
        for batch in batches:
            for record in batch.to_pylist():
                partition = arrow_pickler.loads(record)
                yield from hf_reader.read(partition)

    _orig_reader_format = _DataFrameReader.format

    @wraps(_orig_reader_format)
    def _new_format(self: _DataFrameReader, source: str) -> _DataFrameReader:
        self._format = source
        return _orig_reader_format(self, source)
    
    _DataFrameReader.format = _new_format

    _orig_reader_option = _DataFrameReader.option

    @wraps(_orig_reader_option)
    def _new_option(self: _DataFrameReader, key, value) -> _DataFrameReader:
        if not hasattr(self, "_options"):
            self._options = {}
        self._options[key] = value
        return _orig_reader_option(self, key, value)
    
    _DataFrameReader.option = _new_option

    _orig_reader_options = _DataFrameReader.options

    @wraps(_orig_reader_options)
    def _new_options(self: _DataFrameReader, **options) -> _DataFrameReader:
        if not hasattr(self, "_options"):
            self._options = {}
        self._options.update(options)
        return _orig_reader_options(self, **options)
    
    _DataFrameReader.options = _new_options

    _orig_reader_load = _DataFrameReader.load

    @wraps(_orig_reader_load)
    def _new_load(
        self: _DataFrameReader,
        path: Optional["PathOrPaths"] = None,
        format: Optional[str] = None,
        schema: Optional[Union["StructType", str]] = None,
        **options: "OptionalPrimitiveType",
    ) -> "DataFrame":
        if (format or getattr(self, "_format", None)) == "huggingface":
            from functools import partial
            from pyspark.sql import SparkSession
            from pyspark_huggingface.huggingface import HuggingFaceDatasets
            
            source = HuggingFaceDatasets(options={**getattr(self, "_options", {}), **options, "path": path}).get_source()
            schema = schema or source.schema()
            hf_reader = source.reader(schema)
            partitions = hf_reader.partitions()
            arrow_pickler = _ArrowPickler("partition")
            spark = self._spark if isinstance(self._spark, SparkSession) else self._spark.sparkSession  # _spark is SQLContext for older versions
            rdd = spark.sparkContext.parallelize([arrow_pickler.dumps(partition) for partition in partitions], len(partitions))
            df = spark.createDataFrame(rdd)
            return df.mapInArrow(partial(_read_in_arrow, arrow_pickler=arrow_pickler, hf_reader=hf_reader), schema)
            
        return _orig_reader_load(self, path=path, format=format, schema=schema, **options)

    _DataFrameReader.load = _new_load

    # Writer

    def _write_in_arrow(batches: Iterator["RecordBatch"], arrow_pickler, hf_writer) -> Iterator["RecordBatch"]:
        from pyarrow import RecordBatch

        commit_message = hf_writer.write(batches)
        yield RecordBatch.from_pylist([arrow_pickler.dumps(commit_message)])

    _orig_writer_format = _DataFrameWriter.format

    @wraps(_orig_writer_format)
    def _new_format(self: _DataFrameWriter, source: str) -> _DataFrameWriter:
        self._format = source
        return _orig_writer_format(self, source)
    
    _DataFrameWriter.format = _new_format

    _orig_writer_option = _DataFrameWriter.option

    @wraps(_orig_writer_option)
    def _new_option(self: _DataFrameWriter, key, value) -> _DataFrameWriter:
        if not hasattr(self, "_options"):
            self._options = {}
        self._options[key] = value
        return _orig_writer_option(self, key, value)
    
    _DataFrameWriter.option = _new_option

    _orig_writer_options = _DataFrameWriter.options

    @wraps(_orig_writer_options)
    def _new_options(self: _DataFrameWriter, **options) -> _DataFrameWriter:
        if not hasattr(self, "_options"):
            self._options = {}
        self._options.update(options)
        return _orig_writer_options(self, **options)
    
    _DataFrameWriter.options = _new_options

    _orig_writer_save = _DataFrameWriter.save

    @wraps(_orig_writer_save)
    def _new_save(
        self: _DataFrameWriter,
        path: Optional["PathOrPaths"] = None,
        format: Optional[str] = None,
        mode: Optional[Union["StructType", str]] = None,
        partitionBy: Optional[Union[str, List[str]]] = None,
        **options: "OptionalPrimitiveType",
    ) -> "DataFrame":
        if (format or getattr(self, "_format", None)) == "huggingface":
            from functools import partial
            from pyspark_huggingface.huggingface import HuggingFaceDatasets

            sink = HuggingFaceDatasets(options={**getattr(self, "_options", {}), **options, "path": path}).get_sink()
            schema = self._df.schema
            mode = options.pop("mode", None)
            hf_writer = sink.writer(schema, overwrite=(mode == "overwrite"))
            arrow_pickler = _ArrowPickler("commit_message")
            commit_messages = self._df.mapInArrow(partial(_write_in_arrow, arrow_pickler=arrow_pickler, hf_writer=hf_writer), arrow_pickler.schema).collect()
            commit_messages = [arrow_pickler.loads(commit_message) for commit_message in commit_messages]
            hf_writer.commit(commit_messages)
            return
            
        return _orig_writer_save(self, path=path, format=format, schema=schema, **options)

    _DataFrameWriter.save = _new_save

    # Log only in driver

    if not os.environ.get("SPARK_ENV_LOADED"):
        logging.getLogger(__name__).warning(f"huggingface datasource enabled for pyspark {pyspark.__version__} (backport from pyspark 4)")
