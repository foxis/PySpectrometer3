"""Processing pipeline for chaining spectrum processors."""

from typing import Optional
from ..core.spectrum import SpectrumData
from .base import ProcessorInterface


class ProcessingPipeline:
    """Composable pipeline for processing spectrum data.
    
    The pipeline chains multiple processors together, passing the output
    of each processor to the next. This enables flexible configuration
    of signal processing workflows.
    
    Example:
        pipeline = ProcessingPipeline([
            SavitzkyGolayFilter(window=17, poly=7),
            PeakDetector(min_distance=50, threshold=20),
        ])
        processed = pipeline.run(spectrum_data)
    """
    
    def __init__(self, processors: Optional[list[ProcessorInterface]] = None):
        """Initialize pipeline with optional processors.
        
        Args:
            processors: List of processors to chain together
        """
        self._processors: list[ProcessorInterface] = processors or []
    
    @property
    def processors(self) -> list[ProcessorInterface]:
        """Get list of processors in the pipeline."""
        return self._processors.copy()
    
    def add(self, processor: ProcessorInterface) -> "ProcessingPipeline":
        """Add a processor to the end of the pipeline.
        
        Args:
            processor: Processor to add
            
        Returns:
            Self for method chaining
        """
        self._processors.append(processor)
        return self
    
    def insert(self, index: int, processor: ProcessorInterface) -> "ProcessingPipeline":
        """Insert a processor at a specific position.
        
        Args:
            index: Position to insert at
            processor: Processor to insert
            
        Returns:
            Self for method chaining
        """
        self._processors.insert(index, processor)
        return self
    
    def remove(self, processor: ProcessorInterface) -> "ProcessingPipeline":
        """Remove a processor from the pipeline.
        
        Args:
            processor: Processor to remove
            
        Returns:
            Self for method chaining
        """
        self._processors.remove(processor)
        return self
    
    def clear(self) -> "ProcessingPipeline":
        """Remove all processors from the pipeline.
        
        Returns:
            Self for method chaining
        """
        self._processors.clear()
        return self
    
    def run(self, data: SpectrumData) -> SpectrumData:
        """Run spectrum data through all processors in sequence.
        
        Args:
            data: Input spectrum data
            
        Returns:
            Processed spectrum data
        """
        for processor in self._processors:
            if processor.enabled:
                data = processor.process(data)
        return data
    
    def __len__(self) -> int:
        return len(self._processors)
    
    def __iter__(self):
        return iter(self._processors)
