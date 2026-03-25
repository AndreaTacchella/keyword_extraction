"""checkpoint.py — checkpoint/resume support for long-running scoring jobs.

Manages a directory of partial result files (parquet) and a manifest (JSON)
that tracks which chunks have been completed.  On restart, completed chunks
are skipped and their results are loaded from disk.

Directory layout
----------------
    <checkpoint_dir>/
        manifest.json            # {"completed": ["chunk_0000", ...]}
        chunk_0000.parquet
        chunk_0001.parquet
        ...

Usage
-----
    mgr = CheckpointManager("outputs/checkpoints/semantic")

    for chunk_id, chunk_df in chunks:
        if mgr.is_done(chunk_id):
            continue
        result = ... compute scores for chunk_df ...
        mgr.save(chunk_id, result)

    full_result = mgr.load_all()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages chunk-level checkpointing for a multi-chunk processing job.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory where chunk parquet files and manifest.json are stored.
        Created automatically if it does not exist.
    """

    _MANIFEST = "manifest.json"

    def __init__(self, checkpoint_dir: str | Path):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.dir / self._MANIFEST
        self._completed: set[str] = self._load_manifest()

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self) -> set[str]:
        if self._manifest_path.exists():
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            completed = set(data.get("completed", []))
            if completed:
                logger.info(
                    "Checkpoint: found %d completed chunks in %s",
                    len(completed), self.dir,
                )
            return completed
        return set()

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps({"completed": sorted(self._completed)}, indent=2),
            encoding="utf-8",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_done(self, chunk_id: str) -> bool:
        """Return True if this chunk has already been completed."""
        return chunk_id in self._completed

    def save(self, chunk_id: str, df: pd.DataFrame) -> None:
        """
        Persist a completed chunk and mark it in the manifest.

        Parameters
        ----------
        chunk_id : str
            Unique identifier for this chunk (e.g. "chunk_0000").
        df : pd.DataFrame
            Scored rows for this chunk.
        """
        chunk_path = self.dir / f"{chunk_id}.parquet"
        df.to_parquet(chunk_path, index=False)
        self._completed.add(chunk_id)
        self._save_manifest()
        logger.debug("Checkpoint: saved chunk %s (%d rows) → %s", chunk_id, len(df), chunk_path)

    def load_all(self) -> pd.DataFrame:
        """
        Concatenate all completed chunks into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Rows from all completed chunks, in chunk_id order.
        """
        if not self._completed:
            return pd.DataFrame()

        parts = []
        for chunk_id in sorted(self._completed):
            chunk_path = self.dir / f"{chunk_id}.parquet"
            if not chunk_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint manifest lists '{chunk_id}' but "
                    f"{chunk_path} does not exist. "
                    "Re-run with a clean checkpoint directory."
                )
            parts.append(pd.read_parquet(chunk_path))

        result = pd.concat(parts, ignore_index=True)
        logger.info(
            "Checkpoint: loaded %d rows from %d chunks in %s",
            len(result), len(parts), self.dir,
        )
        return result

    def n_completed(self) -> int:
        """Number of completed chunks."""
        return len(self._completed)

    def clear(self) -> None:
        """
        Delete all checkpoint files and reset the manifest.
        Use this to force a full recomputation.
        """
        for chunk_id in list(self._completed):
            chunk_path = self.dir / f"{chunk_id}.parquet"
            if chunk_path.exists():
                chunk_path.unlink()
        self._completed.clear()
        if self._manifest_path.exists():
            self._manifest_path.unlink()
        logger.info("Checkpoint: cleared all chunks in %s", self.dir)
