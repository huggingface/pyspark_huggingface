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
    from huggingface_hub import CommitOperationAdd
    from pyarrow import RecordBatch

logger = logging.getLogger(__name__)

class HuggingFaceSink(DataSource):
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

        self.path = path
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

    # Get the commit operations to delete all existing Parquet files
    def get_delete_operations(self, resolved_path):
        from huggingface_hub import CommitOperationDelete, list_repo_tree
        from huggingface_hub.hf_api import RepoFile

        # List all files in the directory
        objects = list_repo_tree(
            token=self.token,
            path_in_repo=resolved_path.path_in_repo,
            repo_id=resolved_path.repo_id,
            repo_type=resolved_path.repo_type,
            revision=resolved_path.revision,
            expand=False,
            recursive=False,
        )

        # Delete all existing parquet files
        for obj in objects:
            if isinstance(obj, RepoFile) and obj.path.endswith(".parquet"):
                yield CommitOperationDelete(path_in_repo=obj.path, is_folder=False)

    def write(self, iterator: Iterator["RecordBatch"]) -> HuggingFaceCommitMessage:
        import io

        from huggingface_hub import CommitOperationAdd
        from pyarrow import parquet as pq
        from pyspark import TaskContext
        from pyspark.sql.pandas.types import to_arrow_schema

        # Get the current partition ID
        context = TaskContext.get()
        assert context, "No active Spark task context"
        partition_id = context.partitionId()

        fs = self.get_filesystem()
        resolved_path = fs.resolve_path(self.path)

        schema = to_arrow_schema(self.schema)
        num_files = 0
        additions = []

        with io.BytesIO() as parquet:

            def flush():
                nonlocal num_files
                name = f"{self.uuid}-part-{partition_id}-{num_files}.parquet"  # Name of the file in the repo
                num_files += 1
                parquet.seek(0)

                addition = CommitOperationAdd(
                    path_in_repo=name, path_or_fileobj=parquet
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

            # Write the Parquet files, limiting the size of each file
            while True:
                with pq.ParquetWriter(parquet, schema, **self.kwargs) as writer:
                    num_batches = 0
                    for batch in iterator:
                        writer.write_batch(batch, row_group_size=self.row_group_size)
                        num_batches += 1
                        if parquet.tell() > self.max_bytes_per_file:
                            flush()
                            break
                    else:
                        if num_batches > 0:
                            flush()
                        break

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

        num_commits = math.ceil(len(operations) / self.max_operations_per_commit)
        for i in range(num_commits):
            begin = i * self.max_operations_per_commit
            end = (i + 1) * self.max_operations_per_commit
            operations = operations[begin:end]
            commit_message = "Upload using PySpark" + (
                f" (part {i:05d}-of-{num_commits:05d})" if num_commits > 1 else ""
            )
            fs._api.create_commit(
                repo_id=resolved_path.repo_id,
                repo_type=resolved_path.repo_type,
                revision=resolved_path.revision,
                operations=operations,
                commit_message=commit_message,
            )

    def abort(self, messages: List[HuggingFaceCommitMessage]) -> None:  # type: ignore[override]
        additions = [addition for message in messages for addition in message.additions]
        for addition in additions:
            logger.info(f"Aborted {addition}")
