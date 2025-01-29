import ast
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

from pyspark.sql.datasource import (
    DataSource,
    DataSourceArrowWriter,
    WriterCommitMessage,
)
from pyspark.sql.types import StructType

if TYPE_CHECKING:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from pyarrow import RecordBatch

logger = logging.getLogger(__name__)

class HuggingFaceSink(DataSource):
    """
    A DataSource for writing Spark DataFrames to HuggingFace Datasets.

    This data source allows writing Spark DataFrames to the HuggingFace Hub as Parquet files.

    Name: `huggingfacesink`

    Data Source Options:
    - token (str, required): HuggingFace API token for authentication.
    - path (str, required): HuggingFace repository ID, e.g. `{username}/{dataset}`.
    - path_in_repo (str): Path within the repository to write the data. Defaults to the root.
    - split (str): Split name to write the data to. Defaults to `train`. Only `train`, `test`, and `validation` are supported.
    - revision (str): Branch, tag, or commit to write to. Defaults to the main branch.
    - endpoint (str): Custom HuggingFace API endpoint URL.
    - max_bytes_per_file (int): Maximum size of each Parquet file.
    - row_group_size (int): Row group size when writing Parquet files.
    - max_operations_per_commit (int): Maximum number of files to add/delete per commit.

    Modes:
    - `overwrite`: Overwrite an existing dataset by deleting existing Parquet files.
    - `append`: Append data to an existing dataset.

    Examples
    --------

    Write a DataFrame to the HuggingFace Hub.

    >>> df.write.format("huggingfacesink").mode("overwrite").options(token="...").save("user/dataset")

    Append to an existing dataset on the HuggingFace Hub.

    >>> df.write.format("huggingfacesink").mode("append").options(token="...").save("user/dataset")

    Write to the `test` split of a dataset.

    >>> df.write.format("huggingfacesink").mode("overwrite").options(token="...", split="test").save("user/dataset")
    """

    def __init__(self, options):
        super().__init__(options)

        if "path" not in options or not options["path"]:
            raise Exception("You must specify a dataset name.")

        kwargs = dict(self.options)
        self.token = kwargs.pop("token")
        self.repo_id = kwargs.pop("path")
        self.path_in_repo = kwargs.pop("path_in_repo", None)
        self.split = kwargs.pop("split", None)
        self.revision = kwargs.pop("revision", None)
        self.endpoint = kwargs.pop("endpoint", None)
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
        self.kwargs = kwargs

    @classmethod
    def name(cls):
        return "huggingfacesink"

    def writer(self, schema: StructType, overwrite: bool) -> DataSourceArrowWriter:
        return HuggingFaceDatasetsWriter(
            repo_id=self.repo_id,
            path_in_repo=self.path_in_repo,
            split=self.split,
            revision=self.revision,
            schema=schema,
            overwrite=overwrite,
            token=self.token,
            endpoint=self.endpoint,
            **self.kwargs,
        )


@dataclass
class HuggingFaceCommitMessage(WriterCommitMessage):
    additions: List["CommitOperationAdd"]


class HuggingFaceDatasetsWriter(DataSourceArrowWriter):
    repo_type = "dataset"

    def __init__(
        self,
        *,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        schema: StructType,
        overwrite: bool,
        token: str,
        endpoint: Optional[str] = None,
        row_group_size: Optional[int] = None,
        max_bytes_per_file=500_000_000,
        max_operations_per_commit=100,
        **kwargs,
    ):
        import uuid

        self.repo_id = repo_id
        self.path_in_repo = (path_in_repo or "").strip("/")
        self.split = split or "train"
        self.revision = revision
        self.schema = schema
        self.overwrite = overwrite
        self.token = token
        self.endpoint = endpoint
        self.row_group_size = row_group_size
        self.max_bytes_per_file = max_bytes_per_file
        self.max_operations_per_commit = max_operations_per_commit
        self.kwargs = kwargs

        # Use a unique filename prefix to avoid conflicts with existing files
        self.uuid = uuid.uuid4()

        self.validate()

    def validate(self):
        if self.split not in ["train", "test", "validation"]:
            """
            TODO: Add support for custom splits.

            For custom split names to be recognized, the files must have path with format:
            `data/{split}-{iiiii}-of-{nnnnn}.parquet`
            where `iiiii` is the part number and `nnnnn` is the total number of parts, both padded to 5 digits.
            Example: `data/custom-00000-of-00002.parquet`

            Therefore the current usage of UUID to avoid naming conflicts won't work for custom split names.
            To fix this we can rename the files in the commit phase to satisfy the naming convention.
            """
            raise NotImplementedError(
                f"Only 'train', 'test', and 'validation' splits are supported. Got '{self.split}'."
            )

    def get_api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token, endpoint=self.endpoint)

    @property
    def prefix(self) -> str:
        return os.path.join(self.path_in_repo, self.split)

    def get_delete_operations(self) -> Iterator["CommitOperationDelete"]:
        """
        Get the commit operations to delete all existing Parquet files.
        This is used when `overwrite=True` to clear the target directory.
        """
        from huggingface_hub import CommitOperationDelete
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFolder

        api = self.get_api()

        try:
            objects = api.list_repo_tree(
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                expand=False,
                recursive=False,
            )
            for obj in objects:
                if obj.path.startswith(self.prefix):
                    yield CommitOperationDelete(
                        path_in_repo=obj.path, is_folder=isinstance(obj, RepoFolder)
                    )
        except EntryNotFoundError as e:
            logger.info(f"Writing to a new path: {e}")

    def write(self, iterator: Iterator["RecordBatch"]) -> HuggingFaceCommitMessage:
        import io

        from huggingface_hub import CommitOperationAdd
        from pyarrow import parquet as pq
        from pyspark import TaskContext
        from pyspark.sql.pandas.types import to_arrow_schema

        # Get the current partition ID. Use this to generate unique filenames for each partition.
        context = TaskContext.get()
        partition_id = context.partitionId() if context else 0

        api = self.get_api()

        schema = to_arrow_schema(self.schema)
        num_files = 0
        additions = []

        # TODO: Evaluate the performance of using a temp file instead of an in-memory buffer.
        with io.BytesIO() as parquet:

            def flush(writer: pq.ParquetWriter):
                """
                Upload the current Parquet file and reset the buffer.
                """
                writer.close()  # Close the writer to flush the buffer
                nonlocal num_files
                name = (
                    f"{self.prefix}-{self.uuid}-part-{partition_id}-{num_files}.parquet"
                )
                num_files += 1
                parquet.seek(0)

                addition = CommitOperationAdd(
                    path_in_repo=name, path_or_fileobj=parquet
                )
                api.preupload_lfs_files(
                    repo_id=self.repo_id,
                    additions=[addition],
                    repo_type=self.repo_type,
                    revision=self.revision,
                )
                additions.append(addition)

                # Reuse the buffer for the next file
                parquet.seek(0)
                parquet.truncate()

            """
            Write the Parquet files, flushing the buffer when the file size exceeds the limit.
            Limiting the size is necessary because we are writing them in memory.
            """
            while True:
                with pq.ParquetWriter(parquet, schema, **self.kwargs) as writer:
                    num_batches = 0
                    for batch in iterator:  # Start iterating from where we left off
                        writer.write_batch(batch, row_group_size=self.row_group_size)
                        num_batches += 1
                        if parquet.tell() > self.max_bytes_per_file:
                            flush(writer)
                            break  # Start a new file
                    else:  # Finished writing all batches
                        if num_batches > 0:
                            flush(writer)
                        break  # Exit while loop

        return HuggingFaceCommitMessage(additions=additions)

    def commit(self, messages: List[HuggingFaceCommitMessage]) -> None:  # type: ignore[override]
        import math

        api = self.get_api()
        operations = [
            addition for message in messages for addition in message.additions
        ]
        if self.overwrite:  # Delete existing files if overwrite is enabled
            operations.extend(self.get_delete_operations())

        """
        Split the commit into multiple parts if necessary.
        The HuggingFace API may time out if there are too many operations in a single commit.
        """
        num_commits = math.ceil(len(operations) / self.max_operations_per_commit)
        for i in range(num_commits):
            begin = i * self.max_operations_per_commit
            end = (i + 1) * self.max_operations_per_commit
            part = operations[begin:end]
            commit_message = "Upload using PySpark" + (
                f" (part {i:05d}-of-{num_commits:05d})" if num_commits > 1 else ""
            )
            api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                operations=part,
                commit_message=commit_message,
            )

    def abort(self, messages: List[HuggingFaceCommitMessage]) -> None:  # type: ignore[override]
        # We don't need to do anything here, as the files are not included in the repo until commit
        additions = [addition for message in messages for addition in message.additions]
        for addition in additions:
            logger.info(f"Aborted {addition}")
