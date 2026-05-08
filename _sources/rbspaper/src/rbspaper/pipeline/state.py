"""Pipeline state management for resumable checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
from typing import TypedDict

# ======== Constants ========

STATE_FILENAME = '.pipeline_state.json'


# ======== Pipeline State ========


class PipelineStateDict(TypedDict):
    """Typed dict for deserializing PipelineState from JSON.

    Captures the exact keys and value types produced by :func:`to_dict`
    and consumed by :func:`from_dict`.
    """

    completed: dict[str, list[str]]
    config_hash: str
    started_at: str
    last_updated: str


@dataclass(frozen=True)
class PipelineState:
    """Immutable snapshot of pipeline step completion.

    Tracks which steps have completed, optionally at per-task granularity.
    A step with an empty task list (e.g. {'train': []}) means the step
    completed globally (not tied to a specific downstream task).

    Args:
        completed: Mapping of step name to list of completed task names.
            Empty list means global completion (no task granularity).
        config_hash: 8-character SHA-256 prefix for config integrity.
        started_at: ISO 8601 timestamp of pipeline start.
        last_updated: ISO 8601 timestamp of last state mutation.
    """

    completed: dict[str, list[str]]
    config_hash: str
    started_at: str
    last_updated: str

    def is_step_complete(
        self, *, step: str, task_name: str | None = None
    ) -> bool:
        """Check if a step (optionally for a specific task) is already done.

        Args:
            step: Pipeline step identifier (e.g. 'train', 'encoding').
            task_name: Optional downstream task name for per-task granularity.
                When None, checks global step completion.

        Returns:
            True if the step (and optional task) is marked complete.
        """
        completed_tasks = self.completed.get(step, [])
        if task_name is None:
            return step in self.completed
        return task_name in completed_tasks


# ======== State Builder ========


class _PipelineStateBuilder:
    """Mutable builder for PipelineState snapshots.

    Accumulates step completion markers and produces an immutable
    PipelineState via build().
    """

    def __init__(self, *, config_hash: str) -> None:
        self._completed: dict[str, list[str]] = {}
        self._config_hash = config_hash
        self._started_at = datetime.now(UTC).isoformat()

    def mark_complete(
        self, *, step: str, task_name: str | None = None
    ) -> None:
        """Record that a step (optionally for a task) has finished.

        Args:
            step: Pipeline step identifier.
            task_name: Optional downstream task name. When None, marks
                global completion (empty list in completed dict).
        """
        if step not in self._completed:
            self._completed[step] = []
        if task_name is not None and task_name not in self._completed[step]:
            self._completed[step].append(task_name)
        # task_name is None -> completed[step] stays as [] (global marker)

    def build(self) -> PipelineState:
        """Build an immutable PipelineState from accumulated markers.

        Returns:
            Frozen PipelineState snapshot.
        """
        return PipelineState(
            completed=dict(self._completed),
            config_hash=self._config_hash,
            started_at=self._started_at,
            last_updated=datetime.now(UTC).isoformat(),
        )


# ======== Serialization ========


def to_dict(*, state: PipelineState) -> dict[str, object]:
    """Serialize PipelineState to a JSON-compatible dictionary.

    Args:
        state: PipelineState instance to serialize.

    Returns:
        Dictionary with keys: completed, config_hash, started_at, last_updated.
    """
    return {
        'completed': state.completed,
        'config_hash': state.config_hash,
        'started_at': state.started_at,
        'last_updated': state.last_updated,
    }


def from_dict(*, data: PipelineStateDict) -> PipelineState:
    """Deserialize a dictionary into a PipelineState.

    Args:
        data: Dictionary produced by to_dict or loaded from JSON.

    Returns:
        PipelineState instance.

    Raises:
        KeyError: If required fields are missing.
    """
    return PipelineState(
        completed=data['completed'],
        config_hash=data['config_hash'],
        started_at=data['started_at'],
        last_updated=data['last_updated'],
    )


# ======== Config Hash ========


def compute_config_hash(
    *, model_params: dict[str, object], seed: int
) -> str:
    """Compute an 8-character SHA-256 prefix for config drift detection.

    Hashes a deterministic JSON representation of model parameters and
    seed. Identical inputs always produce the same hash; any change
    in params or seed produces a different hash with probability ~1 - 2^-32.

    Args:
        model_params: Serializable model parameter dictionary. Must not
            contain non-JSON-native types (e.g. LightningModule instances).
        seed: Random seed used for the experiment run.

    Returns:
        8-character lowercase hexadecimal string.
    """
    payload = json.dumps(
        {'params': model_params, 'seed': seed}, sort_keys=True
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]


# ======== Persistence ========


def _json_default(obj: object) -> object:
    """Convert non-JSON-native values for state serialization.

    Handles Path objects by converting to string. State data does not
    include numpy arrays or Enums, so only Path is covered here.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, Path):
        return str(obj)
    message = f"Object of type {type(obj).__name__!r} is not JSON serializable"
    raise TypeError(message)


def _atomic_write_json(*, path: Path, data: dict[str, object]) -> None:
    """Write JSON to disk atomically via .tmp + rename.

    Creates parent directories if they do not exist. Uses
    os.fsync() before rename to ensure data reaches disk.

    Args:
        path: Target file path (e.g. run_dir / '.pipeline_state.json').
        data: Dictionary to serialize as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + '.tmp')
    with tmp_path.open(mode='w', encoding='utf-8') as fh:
        json.dump(obj=data, fp=fh, indent=2, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.rename(path)


def save_pipeline_state(*, state: PipelineState, path: Path) -> None:
    """Persist a PipelineState to disk atomically.

    Writes to path.tmp then atomically renames to path, preventing
    corrupted state files on crash.

    Args:
        state: PipelineState to persist.
        path: Target file path (typically run_dir / STATE_FILENAME).
    """
    data = to_dict(state=state)
    _atomic_write_json(path=path, data=data)


def load_pipeline_state(*, path: Path) -> PipelineState:
    """Load a PipelineState from a JSON file.

    Args:
        path: Path to the state JSON file.

    Returns:
        PipelineState instance reconstructed from the file.

    Raises:
        FileNotFoundError: If the state file does not exist.
    """
    if not path.exists():
        msg = f'State file not found: {path}'
        raise FileNotFoundError(msg)
    with path.open(mode='r', encoding='utf-8') as fh:
        data = json.load(fh)
    return from_dict(data=data)


__all__ = [
    'STATE_FILENAME',
    'PipelineState',
    'PipelineStateDict',
    'compute_config_hash',
    'from_dict',
    'load_pipeline_state',
    'save_pipeline_state',
    'to_dict',
]
