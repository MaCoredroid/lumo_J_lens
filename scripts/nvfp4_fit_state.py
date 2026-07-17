#!/usr/bin/env python3
"""Crash-safe dense-matrix state store for the Qwen3.6 NVFP4 J-lens fit.

Production fitting keeps one little-endian FP32 ``[5120, 5120]`` matrix for
each of source layers 0 through 62.  Current-prompt rows are integrity hashed,
and completed prompts are committed into generation-stamped cumulative sums.
Only JSON and fixed-shape raw numeric files are read; this module never loads
pickle-based state.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STATE_SCHEMA_VERSION = 1
FINAL_ARTIFACT_SCHEMA_VERSION = 1
F32_LE = np.dtype("<f4")
MAX_JSON_BYTES = 16 * 1024 * 1024
STATE_FILENAME = "state.json"
LOCK_FILENAME = ".fit.lock"
CURRENT_DIRECTORY = "current"
FINAL_DIRECTORY = "final-mean"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_value(value: Any) -> Any:
    """Return a detached JSON-only value or reject unsupported objects."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return json.loads(encoded)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    require_regular_file(path, label="hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")


def read_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    require_regular_file(path, label="JSON state")
    size = path.stat().st_size
    if size > max_bytes:
        raise RuntimeError(f"JSON file is too large: {path} ({size} bytes)")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    """Write canonical JSON durably, then atomically replace the destination."""

    normalized = canonical_json_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                normalized,
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class FitLayout:
    """Fixed matrix layout and prompt cardinality bound into resume state."""

    hidden_size: int = 5120
    source_layers: tuple[int, ...] = tuple(range(63))
    prompt_count: int = 10
    io_rows: int = 64

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not self.source_layers:
            raise ValueError("source_layers must not be empty")
        if (
            any(not isinstance(layer, int) or layer < 0 for layer in self.source_layers)
            or len(set(self.source_layers)) != len(self.source_layers)
        ):
            raise ValueError("source_layers must be unique nonnegative integers")
        if self.prompt_count <= 0:
            raise ValueError("prompt_count must be positive")
        if self.io_rows <= 0:
            raise ValueError("io_rows must be positive")

    @property
    def matrix_bytes(self) -> int:
        return self.hidden_size * self.hidden_size * F32_LE.itemsize

    @property
    def matrix_set_bytes(self) -> int:
        return self.matrix_bytes * len(self.source_layers)

    def record(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "source_layers": list(self.source_layers),
            "source_layer_count": len(self.source_layers),
            "prompt_count": self.prompt_count,
            "io_rows": self.io_rows,
            "matrix_dtype": "little-endian-float32",
            "matrix_shape": [self.hidden_size, self.hidden_size],
            "matrix_bytes": self.matrix_bytes,
            "matrix_set_bytes": self.matrix_set_bytes,
        }


PRODUCTION_LAYOUT = FitLayout()


def layer_path(directory: Path, layer: int) -> Path:
    return directory / f"layer-{layer:02d}.f32"


def sum_directory(work_dir: Path, generation: int) -> Path:
    return work_dir / f"sum-{generation:06d}"


def validate_dense_file(path: Path, layout: FitLayout) -> None:
    require_regular_file(path, label="dense matrix")
    actual = path.stat().st_size
    if actual != layout.matrix_bytes:
        raise RuntimeError(
            f"invalid dense matrix file {path}: {actual} != {layout.matrix_bytes}"
        )


def _flush_memmap(matrix: np.memmap, path: Path) -> None:
    matrix.flush()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_matrix_file(path: Path, layout: FitLayout) -> None:
    matrix = np.memmap(
        path,
        mode="w+",
        dtype=F32_LE,
        shape=(layout.hidden_size, layout.hidden_size),
    )
    _flush_memmap(matrix, path)
    del matrix


def open_matrix(path: Path, layout: FitLayout, *, mode: str) -> np.memmap:
    validate_dense_file(path, layout)
    return np.memmap(
        path,
        mode=mode,
        dtype=F32_LE,
        shape=(layout.hidden_size, layout.hidden_size),
    )


def current_chunk_sha256(
    matrices: Sequence[np.memmap],
    source_layers: Sequence[int],
    start: int,
    stop: int,
) -> str:
    if len(matrices) != len(source_layers):
        raise ValueError("matrix count does not match source layers")
    digest = hashlib.sha256()
    for layer, matrix in zip(source_layers, matrices, strict=True):
        digest.update(f"{layer}:{start}:{stop}:<f4\n".encode("ascii"))
        rows = np.asarray(matrix[start:stop], dtype=F32_LE)
        digest.update(rows.tobytes(order="C"))
    return digest.hexdigest()


def validate_current_integrity(
    matrices: Sequence[np.memmap],
    current_state: Mapping[str, Any],
    layout: FitLayout,
) -> None:
    cursor = 0
    chunks = current_state.get("chunks")
    if not isinstance(chunks, list):
        raise RuntimeError("current row-integrity chunks must be a list")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise RuntimeError("current row-integrity chunk must be an object")
        start, stop = chunk.get("start"), chunk.get("stop")
        if (
            start != cursor
            or not isinstance(stop, int)
            or stop <= start
            or stop > layout.hidden_size
        ):
            raise RuntimeError("current row-integrity chunks are not contiguous")
        actual = current_chunk_sha256(
            matrices, layout.source_layers, start, stop
        )
        if actual != chunk.get("sha256"):
            raise RuntimeError(
                f"current row-prefix integrity mismatch at rows {start}:{stop}"
            )
        cursor = stop
    if cursor != current_state.get("next_row"):
        raise RuntimeError("current row-integrity prefix does not reach next_row")


def _sum_records(directory: Path, layout: FitLayout) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer in layout.source_layers:
        path = layer_path(directory, layer)
        validate_dense_file(path, layout)
        records.append(
            {
                "layer": layer,
                "filename": path.name,
                "shape": [layout.hidden_size, layout.hidden_size],
                "dtype": "little-endian-float32",
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_sum_integrity(
    directory: Path,
    state: Mapping[str, Any],
    layout: FitLayout,
) -> None:
    integrity = state.get("sum_integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("generation") != state.get("sum_generation")
    ):
        raise RuntimeError("committed sum generation has no matching integrity manifest")
    records = integrity.get("layers")
    if not isinstance(records, list) or len(records) != len(layout.source_layers):
        raise RuntimeError("committed sum integrity manifest is incomplete")
    for layer, record in zip(layout.source_layers, records, strict=True):
        path = layer_path(directory, layer)
        validate_dense_file(path, layout)
        if (
            not isinstance(record, dict)
            or record.get("layer") != layer
            or record.get("filename") != path.name
            or record.get("size") != path.stat().st_size
        ):
            raise RuntimeError(f"committed sum metadata mismatch at layer {layer}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"committed sum hash mismatch at layer {layer}")
    expected_aggregate = canonical_sha256(records)
    if integrity.get("aggregate_sha256") != expected_aggregate:
        raise RuntimeError("committed sum aggregate hash mismatch")


class FitStateStore:
    """Exclusively locked owner of one resumable dense-fit work directory."""

    def __init__(
        self,
        work_dir: Path,
        layout: FitLayout,
        lock_fd: int,
        state: dict[str, Any],
    ) -> None:
        self.work_dir = work_dir
        self.layout = layout
        self._lock_fd = lock_fd
        self.state = state
        self._closed = False

    @classmethod
    def create(
        cls,
        work_dir: Path,
        contract: Mapping[str, Any],
        *,
        layout: FitLayout = PRODUCTION_LAYOUT,
    ) -> "FitStateStore":
        work_dir = work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = cls._acquire_lock(work_dir)
        try:
            existing = [
                path for path in work_dir.iterdir() if path.name != LOCK_FILENAME
            ]
            if existing:
                raise FileExistsError(
                    "new fit work directory must be empty; resume the existing fit"
                )
            normalized_contract = canonical_json_value(contract)
            now = utc_now()
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "run_id": str(uuid.uuid4()),
                "started_at": now,
                "updated_at": now,
                "status": "running",
                "layout": layout.record(),
                "contract": normalized_contract,
                "contract_sha256": canonical_sha256(normalized_contract),
                "prompt_count": layout.prompt_count,
                "n_done": 0,
                "next_prompt": 0,
                "sum_generation": 0,
                "sum_integrity": None,
                "current": None,
                "committed_prompts": [],
                "final_artifact": None,
            }
            atomic_write_json(work_dir / STATE_FILENAME, state)
            return cls(work_dir, layout, lock_fd, state)
        except Exception:
            cls._release_lock(lock_fd)
            raise

    @classmethod
    def resume(
        cls,
        work_dir: Path,
        contract: Mapping[str, Any],
        *,
        layout: FitLayout = PRODUCTION_LAYOUT,
    ) -> "FitStateStore":
        work_dir = work_dir.expanduser().resolve()
        if not work_dir.is_dir():
            raise FileNotFoundError(f"fit work directory does not exist: {work_dir}")
        lock_fd = cls._acquire_lock(work_dir)
        try:
            state = read_json(work_dir / STATE_FILENAME)
            if not isinstance(state, dict):
                raise RuntimeError("fit state must be a JSON object")
            normalized_contract = canonical_json_value(contract)
            cls._validate_state(
                state,
                layout,
                normalized_contract,
            )
            store = cls(work_dir, layout, lock_fd, state)
            store._recover()
            return store
        except Exception:
            cls._release_lock(lock_fd)
            raise

    @staticmethod
    def _acquire_lock(work_dir: Path) -> int:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(work_dir / LOCK_FILENAME, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError("fit lock must not be a symlink") from exc
            raise
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RuntimeError("fit lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            FitStateStore._release_lock(descriptor)
            raise RuntimeError(f"fit work directory is already locked: {work_dir}") from exc
        return descriptor

    @staticmethod
    def _release_lock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_state(
        state: Mapping[str, Any],
        layout: FitLayout,
        contract: Any,
    ) -> None:
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise RuntimeError("unsupported fit-state schema")
        if state.get("layout") != layout.record():
            raise RuntimeError("resume layout mismatch")
        if (
            state.get("contract") != contract
            or state.get("contract_sha256") != canonical_sha256(contract)
        ):
            raise RuntimeError("resume contract mismatch")
        n_done = state.get("n_done")
        next_prompt = state.get("next_prompt")
        generation = state.get("sum_generation")
        committed = state.get("committed_prompts")
        if (
            not isinstance(n_done, int)
            or not isinstance(next_prompt, int)
            or not isinstance(generation, int)
            or not isinstance(committed, list)
            or n_done != next_prompt
            or n_done != generation
            or n_done != len(committed)
            or not 0 <= n_done <= layout.prompt_count
            or state.get("prompt_count") != layout.prompt_count
        ):
            raise RuntimeError("fit-state prompt counters are inconsistent")
        sum_integrity = state.get("sum_integrity")
        if (generation == 0 and sum_integrity is not None) or (
            generation > 0 and not isinstance(sum_integrity, dict)
        ):
            raise RuntimeError("fit-state cumulative-sum metadata is inconsistent")
        if state.get("status") not in ("running", "completed"):
            raise RuntimeError("fit-state status is invalid")
        for index, record in enumerate(committed):
            if (
                not isinstance(record, dict)
                or record.get("prompt_index") != index
                or record.get("sum_generation") != index + 1
                or not isinstance(record.get("prompt"), dict)
                or record.get("prompt_sha256")
                != canonical_sha256(record.get("prompt"))
                or not _is_sha256(record.get("chunk_manifest_sha256"))
                or not _is_sha256(record.get("sum_aggregate_sha256"))
            ):
                raise RuntimeError("committed prompt metadata is inconsistent")
        current = state.get("current")
        if current is not None:
            if (
                not isinstance(current, dict)
                or current.get("prompt_index") != next_prompt
                or not isinstance(current.get("next_row"), int)
                or not 0 <= current["next_row"] <= layout.hidden_size
                or not isinstance(current.get("prompt"), dict)
                or current.get("prompt_sha256")
                != canonical_sha256(current.get("prompt"))
                or not isinstance(current.get("chunks"), list)
            ):
                raise RuntimeError("fit-state current prompt is inconsistent")
        final = state.get("final_artifact")
        if final is not None and (
            n_done != layout.prompt_count or state.get("status") != "completed"
        ):
            raise RuntimeError("final artifact is bound to an incomplete fit")
        if final is None and state.get("status") == "completed":
            raise RuntimeError("completed fit state has no final artifact")
        if state.get("status") == "completed" and current is not None:
            raise RuntimeError("completed fit state still has a current prompt")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("fit state store is closed")

    def _write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.work_dir / STATE_FILENAME, state)

    def _commit_state(self, state: dict[str, Any]) -> None:
        self._write_state(state)
        self.state = state

    def close(self) -> None:
        if self._closed:
            return
        self._release_lock(self._lock_fd)
        self._closed = True

    def __enter__(self) -> "FitStateStore":
        self._require_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _recover(self) -> None:
        self._require_open()
        for path in self.work_dir.glob(f".{STATE_FILENAME}.tmp.*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                raise RuntimeError(f"unexpected state temporary path: {path}")
        for path in self.work_dir.glob("current.init.*"):
            if path.is_dir():
                shutil.rmtree(path)
        for path in self.work_dir.glob("sum-*.tmp.*"):
            if path.is_dir():
                shutil.rmtree(path)
        for path in self.work_dir.glob("final-mean.tmp.*"):
            if path.is_dir():
                shutil.rmtree(path)

        current_dir = self.work_dir / CURRENT_DIRECTORY
        if self.state["current"] is None:
            if current_dir.exists():
                shutil.rmtree(current_dir)
        else:
            matrices = self.open_current_matrices(mode="r+")
            del matrices

        generation = self.state["sum_generation"]
        referenced_sum = sum_directory(self.work_dir, generation) if generation else None
        if referenced_sum is not None:
            validate_sum_integrity(referenced_sum, self.state, self.layout)
        for path in self.work_dir.glob("sum-[0-9][0-9][0-9][0-9][0-9][0-9]"):
            if path != referenced_sum:
                shutil.rmtree(path)

        final_dir = self.work_dir / FINAL_DIRECTORY
        if self.state["final_artifact"] is not None:
            self.validate_final_artifact()
        elif final_dir.exists():
            metadata = self._validate_final_directory(final_dir)
            next_state = copy.deepcopy(self.state)
            next_state["status"] = "completed"
            next_state["final_artifact"] = self._final_state_record(metadata)
            self._commit_state(next_state)

    def begin_prompt(self, prompt: Mapping[str, Any]) -> None:
        self._require_open()
        if self.state["status"] != "running":
            raise RuntimeError("cannot begin a prompt after fit completion")
        if self.state["current"] is not None:
            raise RuntimeError("a prompt is already in progress")
        if self.state["next_prompt"] >= self.layout.prompt_count:
            raise RuntimeError("every configured prompt is already committed")
        prompt_record = canonical_json_value(prompt)
        if not isinstance(prompt_record, dict):
            raise TypeError("prompt record must be a JSON object")

        current_dir = self.work_dir / CURRENT_DIRECTORY
        if current_dir.exists():
            shutil.rmtree(current_dir)
        temporary = self.work_dir / f"current.init.{os.getpid()}.{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            for layer in self.layout.source_layers:
                _new_matrix_file(layer_path(temporary, layer), self.layout)
            _fsync_directory(temporary)
            os.replace(temporary, current_dir)
            _fsync_directory(self.work_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        next_state = copy.deepcopy(self.state)
        next_state["current"] = {
            "prompt_index": self.state["next_prompt"],
            "prompt": prompt_record,
            "prompt_sha256": canonical_sha256(prompt_record),
            "next_row": 0,
            "chunks": [],
        }
        self._commit_state(next_state)

    def _open_current_matrices(
        self,
        *,
        mode: str,
        validate_integrity: bool,
    ) -> list[np.memmap]:
        self._require_open()
        if self.state["current"] is None:
            raise RuntimeError("no prompt is in progress")
        current_dir = self.work_dir / CURRENT_DIRECTORY
        if not current_dir.is_dir():
            raise RuntimeError("current prompt directory is missing")
        matrices = [
            open_matrix(layer_path(current_dir, layer), self.layout, mode=mode)
            for layer in self.layout.source_layers
        ]
        if validate_integrity:
            validate_current_integrity(matrices, self.state["current"], self.layout)
        return matrices

    def open_current_matrices(self, *, mode: str = "r+") -> list[np.memmap]:
        """Open current matrices after fully rehashing their recorded prefix."""

        return self._open_current_matrices(mode=mode, validate_integrity=True)

    def write_current_chunk(
        self,
        rows_by_layer: Mapping[int, np.ndarray] | Sequence[np.ndarray],
        *,
        start: int | None = None,
    ) -> dict[str, Any]:
        """Write and durably record the next contiguous current-prompt rows."""

        self._require_open()
        current = self.state.get("current")
        if current is None:
            raise RuntimeError("no prompt is in progress")
        if start is None:
            start = current["next_row"]
        if start != current["next_row"]:
            raise RuntimeError("row chunk must start at current next_row")
        if isinstance(rows_by_layer, Mapping):
            if set(rows_by_layer) != set(self.layout.source_layers):
                raise ValueError("row mapping does not match source layers")
            values = [rows_by_layer[layer] for layer in self.layout.source_layers]
        else:
            values = list(rows_by_layer)
            if len(values) != len(self.layout.source_layers):
                raise ValueError("row sequence does not match source layers")
        if not values:
            raise ValueError("row chunk must not be empty")
        first = np.asarray(values[0])
        if first.ndim != 2:
            raise ValueError("row chunks must be rank-two matrices")
        row_count = first.shape[0]
        if row_count <= 0:
            raise ValueError("row chunk must contain at least one row")
        stop = start + row_count
        if stop > self.layout.hidden_size:
            raise ValueError("row chunk exceeds hidden size")
        converted: list[np.ndarray] = []
        for layer, value in zip(self.layout.source_layers, values, strict=True):
            array = np.asarray(value, dtype=F32_LE)
            expected = (row_count, self.layout.hidden_size)
            if array.shape != expected:
                raise ValueError(
                    f"layer {layer} row chunk shape {array.shape} != {expected}"
                )
            if not bool(np.isfinite(array).all()):
                raise FloatingPointError(f"layer {layer} row chunk contains non-finite values")
            converted.append(array)

        # This store has held the exclusive lock since the last validated
        # transition.  Rehashing every earlier chunk here would make a prompt
        # O(number_of_chunks^2); hash only the newly durable rows below.
        matrices = self._open_current_matrices(
            mode="r+", validate_integrity=False
        )
        current_dir = self.work_dir / CURRENT_DIRECTORY
        for layer, matrix, array in zip(
            self.layout.source_layers, matrices, converted, strict=True
        ):
            matrix[start:stop] = array
            _flush_memmap(matrix, layer_path(current_dir, layer))
        chunk = {
            "start": start,
            "stop": stop,
            "sha256": current_chunk_sha256(
                matrices, self.layout.source_layers, start, stop
            ),
        }
        del matrices

        next_state = copy.deepcopy(self.state)
        next_state["current"]["chunks"].append(chunk)
        next_state["current"]["next_row"] = stop
        self._commit_state(next_state)
        return chunk

    def discard_current_prompt(self) -> None:
        """Invalidate all rows for the current prompt after a provenance failure."""

        self._require_open()
        if self.state["status"] != "running":
            raise RuntimeError("cannot discard a prompt after fit completion")
        if self.state["current"] is None:
            raise RuntimeError("no prompt is in progress")

        next_state = copy.deepcopy(self.state)
        next_state["current"] = None
        self._commit_state(next_state)
        current_dir = self.work_dir / CURRENT_DIRECTORY
        if current_dir.exists():
            shutil.rmtree(current_dir)

    def commit_current_prompt(self) -> dict[str, Any]:
        """Atomically advance one complete prompt into cumulative FP32 sums."""

        self._require_open()
        current_state = self.state.get("current")
        if (
            current_state is None
            or current_state.get("next_row") != self.layout.hidden_size
        ):
            raise RuntimeError("cannot commit an incomplete prompt")
        matrices = self.open_current_matrices(mode="r")
        del matrices

        old_generation = self.state["sum_generation"]
        new_generation = old_generation + 1
        if new_generation > self.layout.prompt_count:
            raise RuntimeError("prompt commit exceeds configured prompt count")
        old_dir = (
            sum_directory(self.work_dir, old_generation) if old_generation else None
        )
        current_dir = self.work_dir / CURRENT_DIRECTORY
        final_dir = sum_directory(self.work_dir, new_generation)
        temporary = self.work_dir / (
            f"sum-{new_generation:06d}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temporary.mkdir()
        try:
            for layer in self.layout.source_layers:
                current_matrix = open_matrix(
                    layer_path(current_dir, layer), self.layout, mode="r"
                )
                old_matrix = (
                    open_matrix(layer_path(old_dir, layer), self.layout, mode="r")
                    if old_dir is not None
                    else None
                )
                new_path = layer_path(temporary, layer)
                new_matrix = np.memmap(
                    new_path,
                    mode="w+",
                    dtype=F32_LE,
                    shape=(self.layout.hidden_size, self.layout.hidden_size),
                )
                for row in range(0, self.layout.hidden_size, self.layout.io_rows):
                    stop = min(row + self.layout.io_rows, self.layout.hidden_size)
                    if old_matrix is None:
                        new_matrix[row:stop] = current_matrix[row:stop]
                    else:
                        np.add(
                            old_matrix[row:stop],
                            current_matrix[row:stop],
                            out=new_matrix[row:stop],
                        )
                    if not bool(np.isfinite(new_matrix[row:stop]).all()):
                        raise FloatingPointError(
                            f"cumulative sum became non-finite at layer {layer}"
                        )
                _flush_memmap(new_matrix, new_path)
                del current_matrix, old_matrix, new_matrix
            _fsync_directory(temporary)
            sum_records = _sum_records(temporary, self.layout)
            os.replace(temporary, final_dir)
            _fsync_directory(self.work_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        prompt_commit = {
            "prompt_index": current_state["prompt_index"],
            "prompt": current_state["prompt"],
            "prompt_sha256": current_state["prompt_sha256"],
            "chunk_count": len(current_state["chunks"]),
            "chunk_manifest_sha256": canonical_sha256(current_state["chunks"]),
            "sum_generation": new_generation,
            "sum_aggregate_sha256": canonical_sha256(sum_records),
            "committed_at": utc_now(),
        }
        next_state = copy.deepcopy(self.state)
        next_state["n_done"] = new_generation
        next_state["next_prompt"] = new_generation
        next_state["sum_generation"] = new_generation
        next_state["sum_integrity"] = {
            "generation": new_generation,
            "layers": sum_records,
            "aggregate_sha256": canonical_sha256(sum_records),
        }
        next_state["committed_prompts"].append(prompt_commit)
        next_state["current"] = None
        self._commit_state(next_state)

        if current_dir.exists():
            shutil.rmtree(current_dir)
        if old_dir is not None and old_dir.exists():
            shutil.rmtree(old_dir)
        return prompt_commit

    def _build_final_metadata(
        self,
        layers: list[dict[str, Any]],
        artifact_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized_metadata = canonical_json_value(artifact_metadata)
        if not isinstance(normalized_metadata, dict):
            raise TypeError("artifact metadata must be a JSON object")
        return {
            "schema_version": FINAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "lumo-jlens-dense-fp32-means",
            "created_at": utc_now(),
            "run_id": self.state["run_id"],
            "contract_sha256": self.state["contract_sha256"],
            "layout": self.layout.record(),
            "n_prompts": self.state["n_done"],
            "averaging": "arithmetic mean of cumulative little-endian FP32 sums",
            "layers": layers,
            "layer_aggregate_sha256": canonical_sha256(layers),
            "committed_prompts_sha256": canonical_sha256(
                self.state["committed_prompts"]
            ),
            "metadata": normalized_metadata,
            "metadata_sha256": canonical_sha256(normalized_metadata),
        }

    def finalize_means(
        self,
        artifact_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build and atomically publish the final n-prompt FP32 mean directory."""

        self._require_open()
        normalized_metadata = canonical_json_value(artifact_metadata)
        if self.state["n_done"] != self.layout.prompt_count:
            raise RuntimeError("cannot finalize before every prompt is committed")
        if self.state["current"] is not None:
            raise RuntimeError("cannot finalize while a prompt is in progress")
        sum_dir = sum_directory(self.work_dir, self.state["sum_generation"])
        validate_sum_integrity(sum_dir, self.state, self.layout)
        final_dir = self.work_dir / FINAL_DIRECTORY
        if final_dir.exists():
            metadata = self._validate_final_directory(final_dir)
            if metadata.get("metadata") != normalized_metadata:
                raise RuntimeError("final artifact metadata differs from requested metadata")
            if self.state["final_artifact"] is None:
                next_state = copy.deepcopy(self.state)
                next_state["status"] = "completed"
                next_state["final_artifact"] = self._final_state_record(metadata)
                self._commit_state(next_state)
            return metadata

        temporary = self.work_dir / (
            f"final-mean.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        temporary.mkdir()
        try:
            for layer in self.layout.source_layers:
                summed = open_matrix(
                    layer_path(sum_dir, layer), self.layout, mode="r"
                )
                mean_path = layer_path(temporary, layer)
                mean = np.memmap(
                    mean_path,
                    mode="w+",
                    dtype=F32_LE,
                    shape=(self.layout.hidden_size, self.layout.hidden_size),
                )
                for row in range(0, self.layout.hidden_size, self.layout.io_rows):
                    stop = min(row + self.layout.io_rows, self.layout.hidden_size)
                    np.divide(
                        summed[row:stop],
                        self.state["n_done"],
                        out=mean[row:stop],
                    )
                    if not bool(np.isfinite(mean[row:stop]).all()):
                        raise FloatingPointError(
                            f"final mean became non-finite at layer {layer}"
                        )
                _flush_memmap(mean, mean_path)
                del summed, mean
            layers = _sum_records(temporary, self.layout)
            metadata = self._build_final_metadata(layers, normalized_metadata)
            atomic_write_json(temporary / "metadata.json", metadata)
            _fsync_directory(temporary)
            os.replace(temporary, final_dir)
            _fsync_directory(self.work_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        next_state = copy.deepcopy(self.state)
        next_state["status"] = "completed"
        next_state["final_artifact"] = self._final_state_record(metadata)
        self._commit_state(next_state)
        return metadata

    def _final_state_record(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        metadata_path = self.work_dir / FINAL_DIRECTORY / "metadata.json"
        return {
            "directory": FINAL_DIRECTORY,
            "metadata_path": f"{FINAL_DIRECTORY}/metadata.json",
            "metadata_sha256": sha256_file(metadata_path),
            "layer_aggregate_sha256": metadata["layer_aggregate_sha256"],
            "metadata_payload_sha256": metadata["metadata_sha256"],
            "n_prompts": metadata["n_prompts"],
        }

    def _validate_final_directory(self, directory: Path) -> dict[str, Any]:
        metadata = read_json(directory / "metadata.json")
        if not isinstance(metadata, dict):
            raise RuntimeError("final artifact metadata must be a JSON object")
        if (
            metadata.get("schema_version") != FINAL_ARTIFACT_SCHEMA_VERSION
            or metadata.get("artifact_type") != "lumo-jlens-dense-fp32-means"
            or metadata.get("run_id") != self.state["run_id"]
            or metadata.get("contract_sha256") != self.state["contract_sha256"]
            or metadata.get("layout") != self.layout.record()
            or metadata.get("n_prompts") != self.layout.prompt_count
            or metadata.get("n_prompts") != self.state["n_done"]
            or metadata.get("committed_prompts_sha256")
            != canonical_sha256(self.state["committed_prompts"])
        ):
            raise RuntimeError("final artifact metadata does not match fit state")
        records = metadata.get("layers")
        if not isinstance(records, list) or len(records) != len(self.layout.source_layers):
            raise RuntimeError("final artifact layer manifest is incomplete")
        for layer, record in zip(self.layout.source_layers, records, strict=True):
            path = layer_path(directory, layer)
            validate_dense_file(path, self.layout)
            if (
                not isinstance(record, dict)
                or record.get("layer") != layer
                or record.get("filename") != path.name
                or record.get("size") != path.stat().st_size
                or record.get("sha256") != sha256_file(path)
            ):
                raise RuntimeError(f"final artifact hash mismatch at layer {layer}")
        if metadata.get("layer_aggregate_sha256") != canonical_sha256(records):
            raise RuntimeError("final artifact aggregate hash mismatch")
        if metadata.get("metadata_sha256") != canonical_sha256(metadata.get("metadata")):
            raise RuntimeError("final artifact payload metadata hash mismatch")
        return metadata

    def validate_final_artifact(self) -> dict[str, Any]:
        self._require_open()
        record = self.state.get("final_artifact")
        if not isinstance(record, dict):
            raise RuntimeError("fit state has no final artifact")
        directory = self.work_dir / FINAL_DIRECTORY
        metadata = self._validate_final_directory(directory)
        metadata_path = directory / "metadata.json"
        if (
            record.get("directory") != FINAL_DIRECTORY
            or record.get("metadata_path") != f"{FINAL_DIRECTORY}/metadata.json"
            or record.get("metadata_sha256") != sha256_file(metadata_path)
            or record.get("layer_aggregate_sha256")
            != metadata["layer_aggregate_sha256"]
            or record.get("metadata_payload_sha256") != metadata["metadata_sha256"]
            or record.get("n_prompts") != metadata["n_prompts"]
        ):
            raise RuntimeError("final artifact state binding is invalid")
        return metadata


def matrix_statistics(path: Path, layout: FitLayout) -> dict[str, Any]:
    """Bounded-memory descriptive statistics for one fixed-shape matrix."""

    matrix = open_matrix(path, layout, mode="r")
    finite = 0
    minimum = math.inf
    maximum = -math.inf
    squared_norm = 0.0
    for row in range(0, layout.hidden_size, layout.io_rows):
        chunk = np.asarray(matrix[row : row + layout.io_rows])
        finite += int(np.isfinite(chunk).sum())
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        squared_norm += float(np.square(chunk.astype(np.float64)).sum())
    trace = float(np.trace(matrix, dtype=np.float64))
    del matrix
    if finite != layout.hidden_size * layout.hidden_size:
        raise FloatingPointError(f"non-finite matrix: {path}")
    return {
        "shape": [layout.hidden_size, layout.hidden_size],
        "dtype": "little-endian-float32",
        "finite_count": finite,
        "min": minimum,
        "max": maximum,
        "frobenius_norm": math.sqrt(squared_norm),
        "trace": trace,
        "sha256": sha256_file(path),
    }
