import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional

from pyspark.sql.datasource import (
    DataSource,
    DataSourceArrowWriter,
    WriterCommitMessage,
)
from pyspark.sql.types import StructType

if TYPE_CHECKING:
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        HfFileSystemResolvedPath,
    )
    from pyarrow import RecordBatch

logger = logging.getLogger(__name__)

class HuggingFaceSink(DataSource):
    """
    A DataSource for writing Spark DataFrames to HuggingFace Datasets.

    This data source allows writing Spark DataFrames to the HuggingFace Hub as Parquet files.

    Name: `huggingfacesink`

    Path:
    - The path must be a valid HuggingFace dataset path, e.g. `{user}/{repo}` or `{user}/{repo}/{split}`.
    - A revision can be specified using the `@` symbol, e.g. `{user}/{repo}@{revision}`.

    Data Source Options:
    - token (str, required): HuggingFace API token for authentication.
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

    Append to an existing directory on the HuggingFace Hub.

    >>> df.write.format("huggingfacesink").mode("append").options(token="...").save("user/dataset")

    Write to the `test` split of a dataset.

    >>> df.write.format("huggingfacesink").mode("overwrite").options(token="...").save("user/dataset/test")
    """

    def __init__(self, options):
        super().__init__(options)

        if "path" not in options or not options["path"]:
            raise Exception("You must specify a dataset name.")

        kwargs = dict(self.options)
        self.path = kwargs.pop("path")
        self.token = kwargs.pop("token")
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
            path=self.path,
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
    def __init__(
        self,
        *,
        path: str,
        schema: StructType,
        overwrite: bool,
        token: str,
        endpoint: Optional[str] = None,
        row_group_size: Optional[int] = None,
        max_bytes_per_file=500_000_000,
        max_operations_per_commit=25_000,
        **kwargs,
    ):
        import uuid

        self.path = f"datasets/{path}"
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

    def get_filesystem(self):
        from huggingface_hub import HfFileSystem

        return HfFileSystem(token=self.token, endpoint=self.endpoint)

    def get_delete_operations(
        self, resolved_path: "HfFileSystemResolvedPath"
    ) -> Iterator["CommitOperationDelete"]:
        """
        Get the commit operations to delete all existing Parquet files.
        This is used when `overwrite=True` to clear the target directory.
        """
        from huggingface_hub import CommitOperationDelete
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFile

        fs = self.get_filesystem()

        try:
            objects = fs._api.list_repo_tree(
                path_in_repo=resolved_path.path_in_repo,
                repo_id=resolved_path.repo_id,
                repo_type=resolved_path.repo_type,
                revision=resolved_path.revision,
                expand=False,
                recursive=False,
            )
            for obj in objects:
                if isinstance(obj, RepoFile) and obj.path.endswith(".parquet"):
                    yield CommitOperationDelete(path_in_repo=obj.path, is_folder=False)
        except EntryNotFoundError as e:
            logger.info(f"Writing to a new path: {e}")

    def write(self, iterator: Iterator["RecordBatch"]) -> HuggingFaceCommitMessage:
        import io
        import os

        from huggingface_hub import CommitOperationAdd
        from pyarrow import parquet as pq
        from pyspark import TaskContext
        from pyspark.sql.pandas.types import to_arrow_schema

        # Get the current partition ID. Use this to generate unique filenames for each partition.
        context = TaskContext.get()
        assert context, "No active Spark task context"
        partition_id = context.partitionId()

        fs = self.get_filesystem()
        resolved_path = fs.resolve_path(self.path)

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
                name = f"{self.uuid}-part-{partition_id}-{num_files}.parquet"  # Name of the file in the repo
                path_in_repo = os.path.join(resolved_path.path_in_repo, name)
                num_files += 1
                parquet.seek(0)

                addition = CommitOperationAdd(
                    path_in_repo=path_in_repo, path_or_fileobj=parquet
                )
                fs._api.preupload_lfs_files(
                    repo_id=resolved_path.repo_id,
                    additions=[addition],
                    repo_type=resolved_path.repo_type,
                    revision=resolved_path.revision,
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

        fs = self.get_filesystem()
        resolved_path = fs.resolve_path(self.path)
        operations = [
            addition for message in messages for addition in message.additions
        ]
        if self.overwrite:  # Delete existing files if overwrite is enabled
            operations.extend(self.get_delete_operations(resolved_path))

        """
        Split the commit into multiple parts if necessary.
        The HuggingFace API has a limit of 25,000 operations per commit.
        """
        num_commits = math.ceil(len(operations) / self.max_operations_per_commit)
        for i in range(num_commits):
            begin = i * self.max_operations_per_commit
            end = (i + 1) * self.max_operations_per_commit
            part = operations[begin:end]
            commit_message = "Upload using PySpark" + (
                f" (part {i:05d}-of-{num_commits:05d})" if num_commits > 1 else ""
            )
            fs._api.create_commit(
                repo_id=resolved_path.repo_id,
                repo_type=resolved_path.repo_type,
                revision=resolved_path.revision,
                operations=part,
                commit_message=commit_message,
            )

    def abort(self, messages: List[HuggingFaceCommitMessage]) -> None:  # type: ignore[override]
        # We don't need to do anything here, as the files are not included in the repo until commit
        additions = [addition for message in messages for addition in message.additions]
        for addition in additions:
            logger.info(f"Aborted {addition}")
