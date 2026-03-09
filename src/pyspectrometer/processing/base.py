"""Base interface for spectrum processors."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData


class ProcessorInterface(ABC):
    """Abstract base class for spectrum data processors.

    Processors transform SpectrumData objects, enabling modular
    signal processing pipelines. Each processor performs a single
    transformation (e.g., filtering, peak detection) and returns
    a new SpectrumData with the results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this processor."""
        ...

    @abstractmethod
    def process(self, data: "SpectrumData") -> "SpectrumData":
        """Process spectrum data.

        Args:
            data: Input spectrum data

        Returns:
            Processed spectrum data (may be the same object if no changes)
        """
        ...

    @property
    def enabled(self) -> bool:
        """Check if processor is enabled."""
        return True
