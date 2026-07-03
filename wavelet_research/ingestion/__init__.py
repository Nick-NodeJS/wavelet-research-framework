"""Data ingestion layer for the Wavelet Research Platform."""

from wavelet_research.ingestion.loader import load_ticks
from wavelet_research.ingestion.models import TickFrame

__all__ = ["load_ticks", "TickFrame"]
