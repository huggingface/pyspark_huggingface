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
logger.setLevel(logging.INFO)

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
    addition: Optional["CommitOperationAdd"]


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
        max_operations_per_commit=25000,
        **kwargs,
    ):
        import uuid

        self.path = path
        self.schema = schema
        self.overwrite = overwrite
        self.token = token
        self.endpoint = endpoint
        self.row_group_size = row_group_size
        self.max_operations_per_commit = max_operations_per_commit
        self.kwargs = kwargs

        # Use a unique prefix to avoid conflicts with existing files
        self.prefix = f"{uuid.uuid4()}-"

    def get_filesystem(self):
        from huggingface_hub import HfFileSystem

        return HfFileSystem(token=self.token, endpoint=self.endpoint)

    def get_delete_operations(self, resolved_path):
        from huggingface_hub import CommitOperationDelete, list_repo_tree
        from huggingface_hub.hf_api import RepoFile

        # list all files in the directory
        objects = list_repo_tree(
            token=self.token,
            path_in_repo=resolved_path.path_in_repo,
            repo_id=resolved_path.repo_id,
            repo_type=resolved_path.repo_type,
            revision=resolved_path.revision,
            expand=False,
            recursive=False,
        )

        # delete all parquet files
        for obj in objects:
            if isinstance(obj, RepoFile) and obj.path.endswith(".parquet"):
                yield CommitOperationDelete(path_in_repo=obj.path)

    def write(self, iterator: Iterator["RecordBatch"]) -> HuggingFaceCommitMessage:
        import io

        from huggingface_hub import CommitOperationAdd
        from pyarrow import parquet as pq
        from pyspark import TaskContext
        from pyspark.sql.pandas.types import to_arrow_schema

        context = TaskContext.get()
        assert context, "No active Spark task context"
        partition_id = context.partitionId()

        schema = to_arrow_schema(self.schema)
        parquet = io.BytesIO()
        is_empty = True
        with pq.ParquetWriter(parquet, schema, **self.kwargs) as writer:
            for batch in iterator:
                writer.write_batch(batch, row_group_size=self.row_group_size)
                is_empty = False

        if is_empty:  # Skip empty partitions
            return HuggingFaceCommitMessage(addition=None)

        name = (
            f"{self.prefix}part-{partition_id}.parquet"  # Name of the file in the repo
        )
        parquet.seek(0)
        addition = CommitOperationAdd(path_in_repo=name, path_or_fileobj=parquet)

        fs = self.get_filesystem()
        resolved_path = fs.resolve_path(self.path)
        fs._api.preupload_lfs_files(
            repo_id=resolved_path.repo_id,
            additions=[addition],
            repo_type=resolved_path.repo_type,
            revision=resolved_path.revision,
        )

        logger.info(f"Written {name} with content")
        return HuggingFaceCommitMessage(addition=addition)

    def commit(self, messages: List[HuggingFaceCommitMessage]) -> None:  # type: ignore[override]
        import math

        fs = self.get_filesystem()
        resolved_path = fs.resolve_path(self.path)
        operations = [message.addition for message in messages if message.addition]
        if self.overwrite:
            operations += list(self.get_delete_operations(resolved_path))
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
            for operation in operations:
                logger.info(f"Committed {operation}")

    def abort(self, messages: List[HuggingFaceCommitMessage]) -> None:  # type: ignore[override]
        additions = [message.addition for message in messages if message.addition]
        for addition in additions:
            logger.info(f"Aborted {addition}")
