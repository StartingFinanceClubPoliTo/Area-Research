"""Class-based, parity-preserving Barrick valuation pipeline.

The package is deliberately additive: the accepted CODE-010 modules remain
unchanged and act as the golden master until the refactored runner has passed
the characterization suite.
"""

from .application.pipeline import RefactoredBarrickPipeline

__all__ = ["RefactoredBarrickPipeline"]
