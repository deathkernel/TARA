"""TARA upstream integration layer.

External projects stay external. TARA stores source pins, exposes optional
capability adapters, and validates upstream drift before an update is adopted.
"""

from .registry import UpstreamRegistry, UpstreamSource, UpstreamSnapshot

__all__ = ["UpstreamRegistry", "UpstreamSource", "UpstreamSnapshot"]
