import importlib
import os
import re
import tempfile
import time
from pathlib import Path

from lmms_eval.utils import eval_logger


class LanceVideoBlobResolver:
    def __init__(
        self,
        dataset_uri: str,
        id_column: str,
        blob_column: str,
        cache_dir: Path,
        ext_column: str = "video_ext",
        source_name: str = "Lance",
        video_extensions: tuple[str, ...] = ("mp4", "webm", "mkv", "mov"),
    ):
        try:
            lance = importlib.import_module("lance")
        except ModuleNotFoundError as exc:
            raise ImportError("Lance video resolver requires Python package `pylance` (module import name: `lance`) and `pyarrow`. Install via: uv add pylance pyarrow") from exc

        self._lance = lance
        self._dataset_uri = dataset_uri
        self._id_column = id_column
        self._blob_column = blob_column
        self._ext_column = ext_column
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_name = source_name
        self._video_extensions = tuple(video_extensions)

        self._dataset = self._lance.dataset(self._dataset_uri)

    @staticmethod
    def _escape_sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _safe_cache_stem(video_id: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", video_id)
        if sanitized == "":
            raise ValueError("video_id cannot be empty after sanitization")
        return sanitized

    @staticmethod
    def _read_blob_file(blob_file) -> bytes:
        try:
            return blob_file.read()
        finally:
            close_method = getattr(blob_file, "close", None)
            if callable(close_method):
                close_method()

    def _load_blob_and_ext(self, video_id: str) -> tuple[bytes, str]:
        escaped_video_id = self._escape_sql_literal(video_id)
        table = self._dataset.scanner(
            columns=[self._blob_column, self._ext_column],
            filter=f"{self._id_column} = '{escaped_video_id}'",
            with_row_id=True,
            with_row_address=True,
        ).to_table()
        if table.num_rows == 0:
            raise FileNotFoundError(f"Video ID {video_id} not found in {self._source_name} dataset: {self._dataset_uri}")
        if table.num_rows != 1:
            raise ValueError(f"Video ID {video_id} expected 1 row but found {table.num_rows} rows in {self._source_name} dataset: {self._dataset_uri}")

        blob_obj = table[self._blob_column][0].as_py()
        ext_obj = table[self._ext_column][0].as_py()
        row_id_obj = table["_rowid"][0].as_py() if "_rowid" in table.column_names else None
        row_addr_obj = table["_rowaddr"][0].as_py() if "_rowaddr" in table.column_names else None

        ext = str(ext_obj).strip().lower() if ext_obj is not None else ""
        if ext == "":
            ext = "mp4"

        if isinstance(blob_obj, dict):
            if row_id_obj is not None:
                blob_file = self._dataset.take_blobs(self._blob_column, ids=[int(row_id_obj)])[0]
                return self._read_blob_file(blob_file), ext
            if row_addr_obj is not None:
                blob_file = self._dataset.take_blobs(self._blob_column, addresses=[int(row_addr_obj)])[0]
                return self._read_blob_file(blob_file), ext

            position = blob_obj.get("position")
            if position is None:
                raise TypeError(f"Unsupported {self._source_name} blob descriptor keys: {sorted(blob_obj.keys())}")
            blob_file = self._dataset.take_blobs(self._blob_column, addresses=[int(position)])[0]
            return self._read_blob_file(blob_file), ext

        if isinstance(blob_obj, bytes):
            return blob_obj, ext
        if isinstance(blob_obj, bytearray):
            return bytes(blob_obj), ext
        if isinstance(blob_obj, memoryview):
            return blob_obj.tobytes(), ext
        if hasattr(blob_obj, "read"):
            return blob_obj.read(), ext

        raise TypeError(f"Unsupported {self._source_name} blob type: {type(blob_obj).__name__}")

    def resolve(self, video_id: str) -> str:
        stem = self._safe_cache_stem(video_id)
        for ext in self._video_extensions:
            cached = self._cache_dir / f"{stem}.{ext}"
            if cached.exists():
                return str(cached)

        start_time = time.perf_counter()
        blob, ext = self._load_blob_and_ext(video_id)
        target_path = self._cache_dir / f"{stem}.{ext}"

        with tempfile.NamedTemporaryFile(dir=self._cache_dir, delete=False) as tmp:
            tmp.write(blob)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target_path)

        eval_logger.debug(f"{self._source_name} resolve miss - video_id={video_id}, ext={ext}, bytes={len(blob)}, elapsed_s={time.perf_counter() - start_time:.4f}")
        return str(target_path)
