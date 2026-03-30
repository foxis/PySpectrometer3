"""Resolved reference CSV search paths (no module-level mutable dirs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config


@dataclass(frozen=True)
class ReferenceSearchPaths:
    """Directories to search for CIE / reference CSV files (relative to CWD when not absolute)."""

    raw_dirs: tuple[Path, ...]

    @classmethod
    def default(cls) -> ReferenceSearchPaths:
        """Same defaults as legacy ``data/reference`` and ``output``."""
        return cls((Path("data") / "reference", Path("output")))

    @classmethod
    def from_config(cls, config: Config) -> ReferenceSearchPaths:
        """Build from ``config.export.reference_dirs`` (expanduser per path)."""
        return cls(tuple(Path(p).expanduser() for p in config.export.reference_dirs))

    def resolved(self) -> list[Path]:
        """Resolve each path relative to :func:`Path.cwd` when not absolute."""
        cwd = Path.cwd()
        out: list[Path] = []
        for d in self.raw_dirs:
            p = Path(d)
            out.append((cwd / p) if not p.is_absolute() else p)
        return out
