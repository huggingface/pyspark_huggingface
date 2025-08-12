import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

from pyspark.sql.types import StructType
from pyspark_huggingface.compat.datasource import (
    DataSource,
    DataSourceArrowWriter,
    WriterCommitMessage,
)

if TYPE_CHECKING:
    from huggingface_hub import (
        CommitOperation,
        CommitOperationAdd,
        HfApi,
    )
    from huggingface_hub.hf_api import RepoFile, RepoFolder
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
    - path_in_repo (str): Path within the repository to write the data. Defaults to "data".
    - split (str): Split name to write the data to. Defaults to `train`.
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

        from huggingface_hub import get_token

        kwargs = dict(self.options)
        self.repo_id = kwargs.pop("path")
        self.path_in_repo = kwargs.pop("path_in_repo", None)
        self.split = kwargs.pop("split", None)
        self.revision = kwargs.pop("revision", None)
        self.token = kwargs.pop("token", None) or get_token()
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

    def writer(self, schema: StructType, overwrite: bool) -> "HuggingFaceDatasetsWriter":
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
        self.path_in_repo = (
            path_in_repo.strip("/") if path_in_repo is not None else "data"
        )
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

    def _get_api(self):
        from huggingface_hub import HfApi

        return HfApi(token=self.token, endpoint=self.endpoint, library_name="pyspark_huggingface")

    @property
    def prefix(self) -> str:
        return f"{self.path_in_repo}/{self.split}".strip("/")

    def _list_split(self, api: "HfApi") -> Iterator[Union["RepoFile", "RepoFolder"]]:
        """
        Get all existing files of the current split.
        """
        from huggingface_hub.utils import EntryNotFoundError

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
                    yield obj
        except EntryNotFoundError:
            pass

    def write(self, iterator: Iterator["RecordBatch"]) -> HuggingFaceCommitMessage:
        import io

        from huggingface_hub import CommitOperationAdd
        from pyarrow import parquet as pq
        from pyspark import TaskContext
        from pyspark.sql.pandas.types import to_arrow_schema

        # Get the current partition ID. Use this to generate unique filenames for each partition.
        context = TaskContext.get()
        partition_id = context.partitionId() if context else 0

        api = self._get_api()

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
        """
        Commit the pre-uploaded Parquet files to the HuggingFace Hub, renaming them to match the expected format:
        `{split}-{current:05d}-of-{total:05d}.parquet`.
        Also delete or rename existing files of the split, depending on the mode.
        """

        from huggingface_hub import CommitOperationCopy, CommitOperationDelete
        from huggingface_hub.hf_api import RepoFile, RepoFolder

        api = self._get_api()

        additions = [addition for message in messages for addition in message.additions]
        operations = {}
        count_new = len(additions)
        count_existing = 0

        def format_path(i):
            return f"{self.prefix}-{i:05d}-of-{count_new + count_existing:05d}.parquet"

        def rename(old_path, new_path):
            if old_path != new_path:
                yield CommitOperationCopy(
                    src_path_in_repo=old_path, path_in_repo=new_path
                )
                yield CommitOperationDelete(path_in_repo=old_path)

        # In overwrite mode, delete existing files
        if self.overwrite:
            for obj in self._list_split(api):
                # Delete old file
                operations[obj.path] = CommitOperationDelete(
                    path_in_repo=obj.path, is_folder=isinstance(obj, RepoFolder)
                )
        # In append mode, rename existing files to have the correct total number of parts
        else:
            rename_operations = []
            existing = list(
                obj for obj in self._list_split(api) if isinstance(obj, RepoFile)
            )
            count_existing = len(existing)
            for i, obj in enumerate(existing):
                new_path = format_path(i)
                rename_operations.extend(rename(obj.path, new_path))
            # Rename files in a separate commit to prevent them from being overwritten by new files of the same name
            self._create_commits(
                api,
                operations=rename_operations,
                message="Rename existing files before uploading new files using PySpark",
            )

        # Rename additions, putting them after existing files if any
        for i, addition in enumerate(additions):
            addition.path_in_repo = format_path(i + count_existing)
            # Overwrite the deletion operation if the file already exists
            operations[addition.path_in_repo] = addition

        # Upload the new files
        self._create_commits(
            api,
            operations=list(operations.values()),
            message="Upload using PySpark",
        )

    def _create_commits(
        self, api: "HfApi", operations: List["CommitOperation"], message: str
    ) -> None:
        """
        Split the commit into multiple parts if necessary.
        The HuggingFace API may time out if there are too many operations in a single commit.
        """
        import math

        num_commits = math.ceil(len(operations) / self.max_operations_per_commit)
        for i in range(num_commits):
            begin = i * self.max_operations_per_commit
            end = (i + 1) * self.max_operations_per_commit
            part = operations[begin:end]
            commit_message = message + (
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
