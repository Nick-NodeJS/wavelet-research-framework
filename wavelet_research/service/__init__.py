"""Wavelet Service — local HTTP API for MT5 integration."""

from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig

__all__ = [
    "ServiceConfig",
    "create_app",
]
